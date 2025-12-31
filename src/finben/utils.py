import httpx
import asyncio
import uuid
import re
import tomllib
from typing import Dict

from loguru import logger

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Part,
    TextPart,
    MessageSendParams,
    Message,
    Role,
    SendMessageRequest,
    SendMessageResponse,
)


def get_skills(agent_card: AgentCard) -> list[dict]:
    """
    Returns agent skills as OpenAI compatible tools
    """

    skills = []
    for s in agent_card.skills:
        # Load properties for input params
        properties = {}
        if "properties" in s:
            for k in s.properties.keys():
                properties[k] = s.properties[k]

        # Skills as a function
        function_spec = {
            "name": s.id,
            "description": s.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        # Check which ones are required
        if "required" in s:
            function_spec["parameters"]["required"] = s.required

        skills.append({
            "type": "function",
            "name": s.id,
            "description": s.description,
            "function": function_spec,
            "strict": True,
        })

    return skills

def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """
    The target str contains tags in the format of <tag_name> ... </tag_name>, 
    parse them out and return a dict"""

    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}


def load_agent_card_toml(agent_color: str):
    """
    Loads the agent card associated with a particular color agent
    """
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/agents/{agent_color}.toml", "rb") as f:
        return tomllib.load(f)

async def get_agent_card(url: str) -> AgentCard | None:
    """
    Get the agent card from provided url
    """
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)

    card: AgentCard | None = await resolver.get_agent_card()

    return card


async def wait_agent_ready(url, timeout=10):
    "wait until the A2A server is ready, check by getting the agent card"
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                return True
            else:
                logger.info(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def send_message(
    url, message, task_id=None, context_id: str = None
) -> SendMessageResponse:
    """
    Sends the message
    """
    card = await get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=120.0)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)

    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    response = await client.send_message(request=req)
    return response
