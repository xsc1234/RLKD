# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional
from Prompt import prompt_math, prompt_code, prompt_science,prompt_puzzle
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask
from openai import OpenAI
from trl.extras.vllm_client import VLLMClient,VLLMClient_only_generate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./Your_GSRM_PATH')
vllm_client = VLLMClient_only_generate(
    '0.0.0.0', 29533, connection_timeout=120.0
)


openai_api_key_m = "EMPTY"
openai_api_base_llm_m = "http://0.0.0.0:29564/v1"
client_llm_m = OpenAI(api_key=openai_api_key_m, base_url=openai_api_base_llm_m)


def call_judge_m(instruction,client):
    chat_response = client.chat.completions.create(model="Qwen2.5-7B-Instruct",
                                                   messages=[{"role": "user",  "content": instruction}])
    return chat_response.choices[0].message.content.split(instruction)[-1]

system = """You are a helpful AI Assistant. Decompose the problem into multiple steps according to the thinking process for reasoning. Format the thinking content in the form of [sub-think]-[Query]-[Answer] for each step. When the [Answer] of a certain step gets the answer to the original question, stop reasoning, and the last answer should include a complete answer to the original question.
[sub-think] represents the thinking process. It is necessary to analyze the previous and next steps as detailed as possible, explain in detail why this step is done to lead to the following query, and do not mention the answer of this step.
[Query] represents the problem that needs to be solved in the current step, and analyze it as detailed as possible. Each query contains only one question.
[Answer] represents the answer to the Query. Give the answer to the [Query] directly and briefly without any reasoning steps and redundant statements.
You should split it into more [sub-think]-[Query]-[Answer] with finer granularity, but do not repeat the conditions given in the question."""


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # print('solution is:')
    # print(solution)
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def accuracy_reward_open_math(completions: list[list[dict[str, str]]], answer: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # print('solution is:')
    # print(solution)
    for content, sol in zip(contents, answer):
        gold_parsed = parse(
            '$' + sol + '$',
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


import re

def split_string(s):
    pattern = r'\[(?:sub-think|Query|Answer)\s+\d+\]'
    parts = re.split(f'({pattern})', s)

    result = []
    current_chunk = ''
    for part in parts:
        if not part:
            continue 
        if re.fullmatch(pattern, part):
            if current_chunk:
                result.append(current_chunk)
            current_chunk = part  
        else:
            current_chunk += part 

    if current_chunk:
        result.append(current_chunk)

    filtered_result = [chunk for chunk in result if re.match(pattern, chunk)]
    return filtered_result

def extract_sections(text):

    blocks = split_string(text)
    pattern_think = re.compile(r'\[(sub-think) (\d+)\]')
    pattern_query = re.compile(r'\[(Query) (\d+)\]')
    pattern_answer = re.compile(r'\[(Answer) (\d+)\]')
    sub_think_list = []
    query_list = []
    answer_list = []
    for block in blocks:
        line = block.strip()
        if pattern_think.match(line):
            sub_think_list.append(line)
        elif pattern_query.match(line):
            query_list.append(line)
        elif pattern_answer.match(line):
            answer_list.append(line)
    if len(sub_think_list) == 0:
        return None
    else:
        return {
            'sub_thinks': sub_think_list,
            'queries': query_list,
            'answers': answer_list
        }


import concurrent.futures


def process_item(i, completion_text, generated_content_i, client_llm_m):
    try:
        searchain_output = completion_text
        s_q_a = extract_sections(searchain_output)
        ground_truth_output = generated_content_i
        gt_s_q_a = extract_sections(ground_truth_output)

        if gt_s_q_a is None or s_q_a is None:
            return (i, 0)

        gt_steps = len(gt_s_q_a['sub_thinks'])
        step_reward = min(1, len(s_q_a['sub_thinks']) / gt_steps)
        reward = []

        # Think reward
        for s_i in range(min(gt_steps, len(s_q_a['sub_thinks']))):
            if len(reward) > 0 and reward[-1] == 0:
                reward.append(0)
                continue
            s_think = s_q_a['sub_thinks'][s_i]
            gt_s_think = gt_s_q_a['sub_thinks'][s_i]
            judge_inst = """
# Please judge whether the following two Think Contents are devoted to solving the same problem. Please only answer yes or no! Please only answer yes or no! Please only answer yes or no!
# Think Content 1: {} \nThink Content 2: {}
# Your Answer (yes or no):""".format(gt_s_think, s_think)
            judge = call_judge_m(judge_inst, client=client_llm_m)
            reward.append(1 if 'yes' in judge.lower() else 0)

        # Query reward
        if reward:
            for q_i in range(min(len(gt_s_q_a['queries']), len(s_q_a['queries']))):
                if q_i >= len(reward) or reward[q_i] == 0:
                    continue
                s_query = s_q_a['queries'][q_i]
                gt_s_query = gt_s_q_a['queries'][q_i]
                judge_inst = """
# Please judge whether the following two Contents are devoted to solving the same problem. Please only answer yes or no! Please only answer yes or no! Please only answer yes or no!
# Content 1: {} \nContent 2: {}
# Your Answer (yes or no):""".format(gt_s_query, s_query)
                judge = call_judge_m(judge_inst, client=client_llm_m)
                if 'no' in judge.lower():
                    reward[q_i] *= 0.5

        # Answer reward
        if reward:
            for a_i in range(min(len(gt_s_q_a['answers']), len(s_q_a['answers']))):
                if a_i >= len(reward) or reward[a_i] == 0:
                    continue
                s_answer = s_q_a['answers'][a_i]
                gt_s_answer = gt_s_q_a['answers'][a_i]
                judge_inst = """
# Please judge whether the following two Contents are equally. Please only answer yes or no! Please only answer yes or no! Please only answer yes or no!
# Content 1: {} \nContent 2: {}
# Your Answer (yes or no):""".format(gt_s_answer, s_answer)
                judge = call_judge_m(judge_inst, client=client_llm_m)
                if 'no' in judge.lower():
                    reward[a_i] *= 0.5
        if len(reward) == 0:
            reward_value = 0
        else:
            reward_value = sum(reward)
        #return (i, reward_value + step_reward)
        return (i, reward_value)
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        return (i, 0)





def searchain_reward(completions, domain, problem, generated_content,**kwargs):

    contents = [completion[0]["content"] for completion in completions]
    thinking = [content.split('</think>')[0] for content in contents]
    input_list = []
    for i in range(len(contents)):
        if domain[i] == 'math':
            prompt = prompt_math
        elif domain[i] == 'code':
            prompt = prompt_code
        elif domain[i] == 'puzzle':
            prompt = prompt_puzzle
        else:
            prompt = prompt_science

        instruction = prompt + problem[i] + '\nYou should decompose this reasoning:\n' + thinking[i]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_list.append(text)

    ordered_set_of_prompts = input_list
    completion_ids = vllm_client.generate(
        prompts=ordered_set_of_prompts,
        n=1,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=10000,
        guided_decoding_regex=None,
    )
    # Decode the generated completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    rewards = [0] * len(contents)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_i = {
            executor.submit(
                process_item,
                i,
                completions_text[i],
                generated_content[i],
                client_llm_m
            ): i for i in range(len(contents))
        }

        for future in concurrent.futures.as_completed(future_to_i):
            i = future_to_i[future]
            _, result = future.result()
            rewards[i] = result

            data_temp = {}
            data_temp['ground_truth'] = generated_content[i]
            data_temp['predict_output'] = completions_text[i]
            data_temp['problem'] = problem[i]
            data_temp['domain'] = domain[i]
            data_temp['reward'] = result


    print('rewards {}'.format(rewards))
    return rewards


def searchain_reward_openr1_math(completions, problem, generation,**kwargs):


    contents = [completion[0]["content"] for completion in completions]
    thinking = [content.split('</think>')[0] for content in contents]
    input_list = []
    for i in range(len(contents)):
        prompt = prompt_math
        instruction = prompt + problem[i] + '\nYou should decompose this reasoning:\n' + thinking[i]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_list.append(text)
    ordered_set_of_prompts = input_list
    completion_ids = vllm_client.generate(
        prompts=ordered_set_of_prompts,
        n=1,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=10000,
        guided_decoding_regex=None,
    )
    # Decode the generated completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    gt_resoning_list = []
    for i in range(len(generation)):
        gt_think = generation[i].split('</think>')[0]
        prompt = prompt_math
        instruction = prompt + problem[i] + '\nYou should decompose this reasoning:\n' + gt_think
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        gt_resoning_list.append(text)
    ordered_set_of_prompts = gt_resoning_list
    completion_ids = vllm_client.generate(
        prompts=ordered_set_of_prompts,
        n=1,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=10000,
        guided_decoding_regex=None,
    )
    # Decode the generated completions
    gt_resoning_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    #print('gt_resoning text is : {}'.format(gt_resoning_text))

    rewards = [0] * len(contents)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_i = {
            executor.submit(
                process_item,
                i,
                completions_text[i],
                gt_resoning_text[i],
                client_llm_m
            ): i for i in range(len(contents))
        }

        for future in concurrent.futures.as_completed(future_to_i):
            i = future_to_i[future]
            _, result = future.result()
            rewards[i] = result

            data_temp = {}
            data_temp['ground_truth'] = gt_resoning_text[i]
            data_temp['predict_output'] = completions_text[i]
            data_temp['problem'] = problem[i]
            data_temp['reward'] = result

    print('rewards {}'.format(rewards))
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using Piston+our IOI package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(piston_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    rewards = code_reward(completions, num_parallel=num_parallel, **kwargs)
    BINARY_THRESHOLD = 0.99
    return [1.0 if reward > BINARY_THRESHOLD else 0.0 for reward in rewards]


def code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)
    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "accuracy_open_math": accuracy_reward_open_math,
        "searchain": searchain_reward,
        "searchain_openr1_math": searchain_reward_openr1_math,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(code_reward, num_parallel=script_args.parallel_code_exec_per_proc), code_reward
        ),
        "binary_code": update_wrapper(
            partial(binary_code_reward, num_parallel=script_args.parallel_code_exec_per_proc), binary_code_reward
        ),
        "ioi_code": update_wrapper(
            partial(ioi_code_reward, test_batch_size=script_args.code_eval_test_batch_size), ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
