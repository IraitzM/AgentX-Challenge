"""White agent implementation - the target agent being tested."""

import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

import os
import json
from openai import OpenAI

from finben.config import settings, logger

def prepare_white_agent_card(url):
    """
    Prepares agent card using the A2A objects classes
    """
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Handles user requests and completes tasks",
        tags=["general"],
        examples=[],
    )
    card = AgentCard(
        name="file_agent",
        description="Test agent from file",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class GeneralWhiteAgentExecutor(AgentExecutor):
    """
    Simple white agent executor looking for the response to the task being sent

    NOTE: Hardcoded for testing purposes, uses no external tool
    """

    def __init__(self, tools = None):
        self.ctx_id_to_messages = {}
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )
        self.tools = tools

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        input = context.get_user_input()
        task = context.current_task

        if context.context_id not in self.ctx_id_to_messages:
            # System instructions
            self.ctx_id_to_messages[context.context_id] = [
                {
                "role": "system",
                "content": """"
                    You are a financial assistant providing faithful information regarding the questions posed by the user.
                    Use tools when available to expand your knowledge.
                """
                }
            ]

        messages = self.ctx_id_to_messages[context.context_id]
        # User request
        messages.append(
            {
                "role": "user",
                "content": input,
            }
        )
        response = self.client.chat.completions.create(
            model=settings.WHITE_AGENT_MODEL,
            messages=messages,
            tools=self.tools,
        )
        logger.debug(f"White response {response.to_json()}")
        response_json = json.loads(response.to_json())

        next_message = response_json["choices"][0]["message"]
        messages.append(
            {
                "role": "assistant",
                "content": next_message["content"],
            }
        )

        # If tool need to be called
        if len(next_message["tool_calls"]) > 0:
            logger.debug(f"Tool calls {len(next_message["tool_calls"])}")
            # Green server should respond with tool response
            # TODO: Send tool request to green server
            await event_queue.enqueue_event(
                new_agent_text_message(
                    next_message["content"], context_id=context.context_id
                )
            )
        else:
            await event_queue.enqueue_event(
                new_agent_text_message(
                    next_message["content"], context_id=context.context_id
                )
            )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(host="localhost", port=9002, tools=None):
    """
    Initiates the white agent

    Args:
        host (str, optional): Host. Defaults to "localhost".
        port (int, optional): Port. Defaults to 9002.
    """
    logger.info("Starting white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=GeneralWhiteAgentExecutor(tools),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
