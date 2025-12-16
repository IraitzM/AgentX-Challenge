"""Green agent implementation - manages assessment and evaluation."""

import uvicorn
import tomllib
import json
import time

import os
from statistics import mean
from openai import OpenAI
import pandas as pd

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts

from finben.config import logger
from finben.utils import send_message, parse_tags


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

    def _get_rubric_messages(self, eval_type: str, question:str, received:str, expected:str, criteria:str):
        """
        Considering the types of the rubric returns a message to be used as evaluator

        Args:
            eval_type (str): Between 'correctness' or 'contradiction'
            question (str): Question to be answered
            received (str): Received answer or statement
            expected (str): Expected answer
            criteria (str): Criteria to be assessed

        Returns:
            list[str]: Returns the prompt to be used
        """
        messages = [
            {
                "role": "system",
                "content": """
                    Play the role of a judge evaluating an assignment.
                    Your task is to assess the rightfulness of the provided answer against the expected one.
                    The answer should be a score from 0.0 to 1.0, where:
                    - 0.0 means the criteria is completely not met
                    - 0.5 means the criteria is partially met
                    - 1.0 means the criteria is fully met
                    Use fractional values (e.g., 0.2, 0.7, 0.85) to express degrees of fulfillment.
                    You MUST only respond with a numeric value between 0.0 and 1.0.
                """
            }
        ]

        if eval_type == 'correctness':
            messages.append(
                {
                    "role": "user",
                    "content": f"""
                        Your duty is to assess the correctness of the provided answer according to the criteria we are looking for.

                        Question to be answered was: {question}
                        Provided answer: {received}

                        To what degree (0.0 to 1.0) is the statement "{criteria}" correct according to the provided answer?
                        Use fractional values to express partial correctness. For example:
                        - 0.0 = completely incorrect or not addressed
                        - 0.3-0.5 = partially correct or partially addressed
                        - 0.7-0.9 = mostly correct with minor gaps
                        - 1.0 = completely correct
                    """
                }
            )
        elif eval_type == "contradiction":
            messages.append(
                {
                    "role": "user",
                    "content": f"""
                        Question to be answered was: {question}
                        Provided answer: {received}
                        Evidence: {criteria}

                        To what degree (0.0 to 1.0) is the evidence in contradiction with the provided answer?
                        Use fractional values to express degrees of contradiction. For example:
                        - 0.0 = no contradiction (evidence fully supports the answer)
                        - 0.3-0.5 = minor contradiction or partial inconsistency
                        - 0.7-0.9 = significant contradiction
                        - 1.0 = complete contradiction
                    """
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"""
                        Question to be answered was: {question}
                        Provided answer: {received}
                        Expected: {expected}

                        To what degree (0.0 to 1.0) do the expected and provided answers overlap?
                        Use fractional values to express similarity. For example:
                        - 0.0 = completely different, no overlap
                        - 0.2-0.4 = minimal overlap, different meaning
                        - 0.5-0.7 = moderate overlap, similar concepts but different wording
                        - 0.8-0.9 = high overlap, very similar meaning
                        - 1.0 = word-by-word coincidence or practically identical meaning
                    """
                }
            )

        return messages


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
        dataset_path = env_config["task_path"]
        finben_data = pd.read_csv(dataset_path)
        logger.info("Green agent: Starting evaluation...")
        metrics = {
            "time_used" : [],
            "time_drift" : [],
            "rubric" : []
        }
        for task_index in env_config["task_ids"]:
            timestamp_started = time.time()
            # Launch
            received = await ask_agent_to_solve(white_agent_url, dataset_path, task_index)

            # Evaluate the response according to the dataset
            time_taken = time.time() - timestamp_started
            metrics["time_used"].append(time_taken)

            # Select for id in the file
            selected = finben_data.iloc[task_index, :]

            question = selected["Question"]
            answer = selected["Answer"]
            expected_time = int(selected["Expert time (mins)"])
            rubric_json = json.loads(selected["Rubric"].replace("'","\"")) # To avoid issues

            metrics["time_drift"].append(expected_time - time_taken)

            # Per operation
            for operation in rubric_json:
                logger.debug(f"Evaluating rubric - Question: {question[:100]}...")
                logger.debug(f"  Expected Answer: {answer[:100]}...")
                logger.debug(f"  Received Answer: {received[:100]}...")
                logger.debug(f"  Operator: {operation['operator']}")
                logger.debug(f"  Criteria: {operation['criteria'][:200]}...")
                response = self.client.chat.completions.create(
                    model=env_config["user_model"],
                    messages=self._get_rubric_messages(
                        eval_type=operation["operator"],
                        question=question,
                        received=received,
                        expected=answer,
                        criteria=operation["criteria"])
                )
                dict_answer = response.to_dict()
                score = dict_answer["choices"][0]["message"]["content"]
                logger.debug(f"  Score: {score}")
                metrics["rubric"].append(float(score))

            # Extra to just check similarity of the answer
            logger.debug(f"Evaluating similarity - Question: {question[:100]}...")
            logger.debug(f"  Expected Answer: {answer[:100]}...")
            logger.debug(f"  Received Answer: {received[:100]}...")
            logger.debug(f"  Operator: similarity")
            response = self.client.chat.completions.create(
                model=env_config["user_model"],
                messages=self._get_rubric_messages(
                    eval_type="similarity",
                    question=question,
                    received=received,
                    expected=answer,
                    criteria="similarity")
            )
            dict_answer = response.to_dict()
            score = dict_answer["choices"][0]["message"]["content"]
            logger.debug(f"  Score: {score}")
            metrics["rubric"].append(float(score))

        # Average scores
        metrics["avg. score"] = mean(metrics["rubric"])
        metrics["task_ids"] = env_config["task_ids"]

        logger.info("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(f"Finished. \n {metrics}\n")
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
