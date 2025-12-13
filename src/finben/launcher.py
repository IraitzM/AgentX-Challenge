"""
Main entrypoint launching the agentified benchamrk
"""
import os
import asyncclick as click
import json
import multiprocessing

from finben.utils import wait_agent_ready, send_message
from finben.agents import start_green_agent, start_white_agent

from loguru import logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Set default log level to INFO
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    level=log_level
)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=True,
    epilog="Specify one of these sub-commands and you can find more help from there.",
)
@click.pass_context
def cli(ctx, **kwargs):
    """
    Finance Benchmarking Tool
    """

@cli.command("run")
async def run():
    """Runs the main benchmark agent suite."""
    logger.info("Running the benchmark agent suite...")

    logger.info("Launching green agent...")
    green_address = ("localhost", os.getenv("GREEN_AGENT_PORT", 9001))
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("green", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    logger.info("Green agent is ready.")

    # start white agent
    logger.info("Launching white agent...")
    white_address = ("localhost", os.getenv("WHITE_AGENT_PORT", 9002))
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(target=start_white_agent, args=white_address)
    p_white.start()
    assert await wait_agent_ready(white_url), "White agent not ready in time"
    logger.info("White agent is ready.")

    # send the task description
    logger.info("Sending task description to green agent...")
    task_config = {
        "env": "retail",
        "user_strategy": "llm",
        "user_model": "moonshotai/Kimi-K2-Instruct",
        "user_provider": "nebius",
        "task_split": "test",
        "task_path": "assets/data/public.csv",
        "task_ids": [1, 10],
    }
    task_text = f"""
        Your task is to instantiate the finance benchmark to test the agent located at:
        <white_agent_url>
        http://{white_address[0]}:{white_address[1]}/
        </white_agent_url>
        You should use the following env configuration:
        <env_config>
        {json.dumps(task_config, indent=2)}
        </env_config>
    """
    logger.info("Task description:")
    logger.info(task_text)
    logger.info("Sending...")
    response = await send_message(green_url, task_text)
    logger.info("Response from green agent:")
    logger.info(response)

    logger.info("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    logger.info("Agents terminated.")

    logger.info("Benchmarking completed.")


if __name__ == "__main__":
    cli()
