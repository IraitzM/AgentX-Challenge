"""Green agent implementation - manages assessment and evaluation."""

import uvicorn
import tomllib
import json
import time

import os
from openai import OpenAI
import pandas as pd

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts

from finben.utils import send_message, parse_tags

from loguru import logger

import dotenv

dotenv.load_dotenv()


def load_agent_card_toml(agent_color: str):
    """
    Loads the agent card associated with a particular color agent
    """
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_color}.toml", "rb") as f:
        return tomllib.load(f)


async def ask_agent_to_solve(white_agent_url, dataset_path, task_index):
    """
    Asks the white agent to solve a given task
    """

    # Prepare the initial message to the white agent
    context_id = None

    # Select for id in the file
    finben_data = pd.read_csv(dataset_path)
    selected = finben_data.iloc[task_index, :]

    task_description = selected["Question"]

    logger.info(
        f"@@@ Green agent: Sending message to white agent{'ctx_id=' + str(context_id) if context_id else ''}... -->\n{task_description}"
    )
    white_agent_response = await send_message(
        white_agent_url, task_description, context_id=context_id
    )
    res_root = white_agent_response.root
    assert isinstance(res_root, SendMessageSuccessResponse)
    res_result = res_root.result
    assert isinstance(
        res_result, Message
    )  # though, a robust design should also support Task
    if context_id is None:
        context_id = res_result.context_id
    else:
        assert context_id == res_result.context_id, (
            "Context ID should remain the same in a conversation"
        )

    text_parts = get_text_parts(res_result.parts)
    assert len(text_parts) == 1, "Expecting exactly one text part from the white agent"
    white_text = text_parts[0]
    logger.info(f"@@@ White agent response:\n{white_text}")

    return white_text


class GreenAgentExecutor(AgentExecutor):
    """
    Main green agent executor class
    """
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        logger.info("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        env_config_str = tags["env_config"]
        env_config = json.loads(env_config_str)

        # set up the environment
        logger.info("Green agent: Setting up the environment...")
        assert len(env_config["task_ids"]) == 1, (
            "Only single task supported for demo purpose"
        )
        task_index = env_config["task_ids"][0]
        dataset_path = env_config["task_path"]

        logger.info("Green agent: Starting evaluation...")
        timestamp_started = time.time()
        received = await ask_agent_to_solve(white_agent_url, dataset_path, task_index)

        # Evaluate the response according to the dataset
        metrics = {}
        metrics["time_used"] = time.time() - timestamp_started

        # Select for id in the file
        finben_data = pd.read_csv(dataset_path)
        selected = finben_data.iloc[task_index, :]

        question = selected["Question"]
        answer = selected["Answer"]
        expected_time = int(selected["Expert time (mins)"])
        rubric_json = json.loads(selected["Rubric"].replace("'","\"")) # To avoid issues

        metrics["time_drift"] = expected_time - metrics["time_used"]

        metrics["rubric"] = [] # Per operation
        for operation in rubric_json:
            response = self.client.chat.completions.create(
                model=env_config["user_model"],
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                            Your task is to check the {operation["operator"]} considering
                            the provided question, received and expected answer.
                        """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                Question: {question}

                                Received answer: {received}
                                Expected answer: {answer}
                                """
                            }
                        ]
                    }
                ]
            )
            metrics["rubric"].append(response.to_json())

        logger.info("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(f"Finished. \nMetrics: {metrics}\n")
        )  # alternative, impl as a task-generating agent

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="green_agent", host="localhost", port=9001):
    """
    Initiates green agent
    """
    logger.info("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
