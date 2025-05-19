import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify


LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d", "e", "f"]
problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def get_query_from_query(q: str):
    try:
        matches = re.findall(problem_pattern, q, re.DOTALL)
        return matches[0]
    except:
        return q


def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    return None


def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response != None:
        response = response
    else:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]
    content, sol = response, answer
    answer_parsed = content
    gold_parsed = parse(sol,
                        extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],)
    
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                answer_parsed = parse(
                    student_answer,
                    extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
                )
                reward = float(verify(student_answer, gold_parsed))
                
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed


def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def format_reward_func_mm(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?=(?:.*<observe>){1})(?=(?:.*<\/observe>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r"(?!.*<observe>.*<observe>)"
        r"(?!.*<\/observe>.*<\/observe>)"
        r".*<observe>(.+?)<\/observe>\s*<think>(.+?)</think>\s*<answer>.+?</answer>.*|.*<think>(.+?)<observe>(.+?)<\/observe>(.+?)<\/think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def reward_func(queries, prompts, labels):
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, answer in zip(queries, prompts, labels):
            try:
                response = get_response_from_query(query)
                if response == "":
                    f.write("Error: " + query + "\n")
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    format_rewards.append(0.0)

                else:
                    query1 = get_query_from_query(query)

                    accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func_mm(response)

                    rewards.append(accuracy_reward + format_reward)
                    accuracy_rewards.append(accuracy_reward)
                    format_rewards.append(format_reward)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1 + "\n")
                    f.write("Response: " + response + "\n")
                    f.write(f"Parsed Answer: {answer_parsed}\tGT Answer: {answer}\n")
                    f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)
                
    return torch.tensor(rewards, dtype=torch.float32)
