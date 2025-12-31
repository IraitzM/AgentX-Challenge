"""
Main entrypoint launching the agentified benchamrk
"""
import multiprocessing
import asyncclick as click

from finben.utils import wait_agent_ready, send_message, get_agent_card, get_skills
from finben.agents import start_green_agent, start_white_agent

from finben.config import logger, settings


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

@cli.command("green")
async def deploy_green_agent():
    """Runs the main benchmark agent suite."""
    logger.info("Running the benchmark agent suite...")

    logger.info("Launching green agent...")
    green_address = (settings.GREEN_AGENT_HOST, settings.GREEN_AGENT_PORT)
    green_url = settings.green_url()
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("green", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    logger.info("Green agent is ready.")

@cli.command("run")
async def run():
    """Runs the main benchmark agent suite."""
    logger.info("Running the benchmark agent suite...")

    logger.info("Launching green agent...")
    green_address = (settings.GREEN_AGENT_HOST, settings.GREEN_AGENT_PORT)
    green_url = settings.green_url()
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("green", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    logger.info("Green agent is ready.")

    # Get exposed tools
    green_agent_card = await get_agent_card(green_url)
    if green_agent_card is None:
        raise RuntimeError("Failed to fetch green agent card")

    # List available functions
    tools = get_skills(green_agent_card)

    # start white agent
    logger.info("Launching white agent...")
    white_address = (settings.WHITE_AGENT_HOST, settings.WHITE_AGENT_PORT)
    white_url = settings.white_url()
    p_white = multiprocessing.Process(
        target=start_white_agent,
        args=[*white_address, tools]
    )
    p_white.start()
    assert await wait_agent_ready(white_url), "White agent not ready in time"
    logger.info("White agent is ready.")

    # send the task description
    logger.info("Sending task description to green agent...")
    logger.debug("Task description:")
    logger.debug(settings.TASK_TEXT)
    logger.info("Sending...")
    response = await send_message(green_url, settings.TASK_TEXT)
    logger.debug("Response from green agent:")
    logger.debug(response)

    logger.info("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    logger.info("Agents terminated.")

    logger.info("Benchmarking completed.")


if __name__ == "__main__":
    cli()
