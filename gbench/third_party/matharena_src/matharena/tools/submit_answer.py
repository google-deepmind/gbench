"""A tool for submitting answers to Project Euler problems."""

import csv
import json
import os
import time
import hashlib
import base64


from loguru import logger


def check_hash_match(ans, target):
    try:
        ANS = int(ans)
    except ValueError:
        return False

    # Giant salt (hundreds of KB or MB)
    SALT = b"A" * (5 * 1024 * 1024)  # 5 MB salt

    def expensive_sha256(value, rounds):
        data = SALT + str(value).encode()
        h = hashlib.sha256(data).digest()
        for _ in range(rounds):
            h = hashlib.sha256(h).digest()
        return base64.b64encode(h).decode()

    G = expensive_sha256(ANS, 1_000_000)
    if G == target:
        return True
    return False


def load_wrong_answers(known_wrong_answers_path: str) -> dict[int, list[str]]:
    known_wrong_answers = {}
    with open(known_wrong_answers_path, "r") as f:
        known_wrong_answers = json.load(f)
        known_wrong_answers = {int(k) + 942: v for k, v in known_wrong_answers.items()}
    return known_wrong_answers


def submit_answer(problem_id: int, answer: str, answers_path="data/euler/euler/answers.csv") -> str:
    """Submits an answer to Project Euler and returns the response.

    Args:
        problem_id (int): The Project Euler problem ID (from the website, not internal, offset is 942).
        answer (str): The answer to submit.
    Returns:
        str: The response from the submission, either "CORRECT" or "WRONG".
    """
    known_wrong_answers_path = answers_path.replace("answers.csv", "known_wrong_answers.json")

    answers = {}
    with open(answers_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_val = int(row["id"].strip()) + 942
            correct_answer = str(row["answer"].strip())
            answers[id_val] = correct_answer

    if problem_id not in answers:
        return f"UNKNOWN PROBLEM ID: {problem_id}, CANNOT SUBMIT ANSWER"

    if answers[problem_id] != "none":
        if "hash:" in answers[problem_id]:
            hval = answers[problem_id].split(":")[1]
            if check_hash_match(str(answer), hval):
                return "CORRECT"
            else:
                return "WRONG"
        else:
            if str(answers[problem_id]) == str(answer):
                return "CORRECT"
            else:
                return "WRONG"

    # Is it in the wrong answers?
    known_wrong_answers = load_wrong_answers(known_wrong_answers_path)
    if problem_id in known_wrong_answers:
        if str(answer) in known_wrong_answers[problem_id]:
            return "WRONG"

    # Use human if answer is unknown
    # Generate random UID for this
    uid = str(int(time.time() * 1000))[-6:]
    os.makedirs("human_toolcalls", exist_ok=True)
    filename = f"human_toolcalls/submit_answer_{problem_id}_{uid}.txt"
    with open(filename, "w") as f:
        f.write(f"Problem ID: {problem_id}\n")
        f.write(f"https://projecteuler.net/problem={problem_id}\n")
        f.write(f"https://projecteuler.info/problem={problem_id}\n")
        f.write(f"Submitted Answer: {answer}\n")
        f.write("Please check and write 1 if CORRECT or 0 if WRONG on the next line.\n")
    orig_lines = 5
    logger.info(f"Submitted answer for problem {problem_id} to {filename} for human review.")
    i = 0
    while True:
        i += 1
        time.sleep(30)  # Poll every 30 seconds
        with open(filename, "r") as f:
            lines = f.readlines()
            if len(lines) >= orig_lines + 1:
                response = lines[orig_lines].strip()
                if response in ["1", "0"]:
                    # FOUND IT!
                    logger.info(f"Found a valid response in {filename}.")
                    os.remove(filename)
                    if response == "0":
                        # Add to known wrong answers
                        known_wrong_answers = load_wrong_answers(known_wrong_answers_path)
                        if problem_id not in known_wrong_answers:
                            known_wrong_answers[problem_id] = []
                        known_wrong_answers[problem_id].append(str(answer))
                        logger.info(
                            f"Added to known wrong answers for problem {problem_id}: {answer} ({known_wrong_answers_path})"
                        )
                        with open(known_wrong_answers_path, "w") as wf:
                            json.dump({k - 942: v for k, v in known_wrong_answers.items()}, wf, indent=4)
                        return "WRONG"
                    else:
                        return "CORRECT"

        if i % 10 == 0:
            logger.debug(f"Did not find a valid response in {filename}, checking again in 30 seconds.")
