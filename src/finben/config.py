import os
import json
from dataclasses import dataclass

from dotenv import load_dotenv, find_dotenv
from loguru import logger as _logger

# Load .env once when this module is imported
load_dotenv(find_dotenv(), override=True)


@dataclass(frozen=True)
class Settings:
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    GREEN_AGENT_HOST: str = os.getenv("GREEN_AGENT_HOST", "localhost")
    GREEN_AGENT_PORT: int = int(os.getenv("GREEN_AGENT_PORT", "9001"))
    WHITE_AGENT_HOST: str = os.getenv("WHITE_AGENT_HOST", "localhost")
    WHITE_AGENT_PORT: int = int(os.getenv("WHITE_AGENT_PORT", "9002"))
    TASK_CONFIG = {
        "env": "retail",
        "user_strategy": "llm",
        "user_model": "moonshotai/Kimi-K2-Instruct",
        "user_provider": "nebius",
        "task_split": "test",
        "task_path": "assets/data/public.csv",
        "task_ids": [1, 10],
    }
    TASK_TEXT = f"""
        Your task is to instantiate the finance benchmark to test the agent located at:
        <white_agent_url>
        http://{WHITE_AGENT_HOST}:{WHITE_AGENT_PORT}/
        </white_agent_url>
        You should use the following env configuration:
        <env_config>
        {json.dumps(TASK_CONFIG, indent=2)}
        </env_config>

        Available tools:
        - google_search_tool: Allows to search for up to date information
    """

    def green_url(self) -> str:
        """Composes green agent url

        Returns:
            str: url
        """
        return f"http://{self.GREEN_AGENT_HOST}:{self.GREEN_AGENT_PORT}"

    def white_url(self) -> str:
        """Composes white agent url, testing purposes

        Returns:
            str: url
        """
        return f"http://{self.WHITE_AGENT_HOST}:{self.WHITE_AGENT_PORT}"


settings = Settings()


def configure_logger():
    """
    Configure logger to be used
    """
    # idempotent config: remove existing handlers and add one handler
    _logger.remove()
    _logger.add(
        lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
    )


# configure logger on import so other modules can `from finben.config import logger`
configure_logger()
logger = _logger
