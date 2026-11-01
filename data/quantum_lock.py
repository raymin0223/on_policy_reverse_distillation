"""Dependency-free port of Reasoning Gym's official ``quantum_lock`` task.

Source:
https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/graphs/quantum_lock.py
"""

from __future__ import annotations

from collections import deque
from random import Random
from typing import Any


QUESTION_TEMPLATE = """\
In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: {initial_value} ({initial_state})
Target: {target_value}
Buttons:
{buttons}"""


def _generate_puzzle(rng: Random, difficulty: int) -> dict[str, Any]:
    base_values = {
        "add": [2, 3] if difficulty >= 5 else [1, 2],
        "subtract": [2, 3] if difficulty >= 5 else [1, 2],
        "multiply": [2, 3] if difficulty >= 7 else [2],
    }
    operations = [
        {"type": "add", "values": base_values["add"]},
        {"type": "subtract", "values": base_values["subtract"]},
        {"type": "multiply", "values": base_values["multiply"]},
    ]

    buttons: list[dict[str, Any]] = []
    used_combinations: set[tuple[str, int, str]] = set()
    while len(buttons) < 3:
        operation = rng.choice(operations)
        value = rng.choice(operation["values"])
        state_weights = {"any": 4, "green": 2, "red": 1}
        active_state = rng.choices(list(state_weights), weights=state_weights.values(), k=1)[0]
        combination = (operation["type"], value, active_state)
        if combination in used_combinations:
            continue
        buttons.append(
            {
                "name": chr(65 + len(buttons)),
                "type": operation["type"],
                "value": value,
                "active_state": active_state,
            }
        )
        used_combinations.add(combination)

    base_target = 5 + difficulty**1.5
    variance = rng.randint(-int(base_target * 0.2), int(base_target * 0.3))
    target = max(8, int(base_target + variance))
    max_steps = min(15, 6 + int(difficulty * 1.5))

    queue = deque([(0, "red", [])])
    visited: set[tuple[int, str]] = set()
    while queue:
        value, state, path = queue.popleft()
        if value == target:
            return {
                "initial_value": 0,
                "initial_state": "red",
                "target_value": target,
                "buttons": buttons,
                "max_steps": max_steps,
                "solution": path,
            }
        if len(path) >= max_steps or (value, state) in visited:
            continue
        visited.add((value, state))
        for button in buttons:
            if button["active_state"] not in (state, "any"):
                continue
            next_state = "green" if state == "red" else "red"
            if button["type"] == "add":
                next_value = value + button["value"]
            elif button["type"] == "subtract":
                next_value = value - button["value"]
            else:
                next_value = value * button["value"]
            queue.append((next_value, next_state, path + [button["name"]]))

    return _generate_puzzle(rng, difficulty)


def generate_quantum_lock_problem(index: int, seed: int, difficulty: int = 10) -> dict[str, Any]:
    if difficulty <= 0:
        raise ValueError("difficulty must be positive")
    rng = Random(seed + index)
    sampled_difficulty = rng.randint(1, difficulty)
    puzzle = _generate_puzzle(rng, sampled_difficulty)
    question = QUESTION_TEMPLATE.format(
        initial_value=puzzle["initial_value"],
        initial_state=puzzle["initial_state"],
        target_value=puzzle["target_value"],
        buttons="\n".join(
            f"{button['name']}: {button['type'].title()} {button['value']} (when {button['active_state']})"
            for button in puzzle["buttons"]
        ),
    )
    return {
        "question": question,
        "answer": " → ".join(puzzle["solution"]),
        "solution_path": puzzle["solution"],
        "target_value": puzzle["target_value"],
        "buttons": puzzle["buttons"],
        "initial_state": puzzle["initial_state"],
        "initial_value": puzzle["initial_value"],
        "difficulty": sampled_difficulty,
    }


def generate_quantum_lock_problems(
    num_problems: int,
    seed: int,
    difficulty: int = 10,
) -> list[dict[str, Any]]:
    if num_problems <= 0:
        raise ValueError("num_problems must be positive")
    return [generate_quantum_lock_problem(index, seed, difficulty) for index in range(num_problems)]


def generate_unique_quantum_lock_problems(
    num_problems: int,
    seed: int,
    difficulty: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep official samples unchanged while removing repeated questions."""

    if num_problems <= 0:
        raise ValueError("num_problems must be positive")
    problems: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    source_index = 0
    max_source_indices = max(num_problems * 20, 10000)
    while len(problems) < num_problems and source_index < max_source_indices:
        problem = generate_quantum_lock_problem(source_index, seed, difficulty)
        problem["source_index"] = source_index
        source_index += 1
        if problem["question"] in seen_questions:
            continue
        seen_questions.add(problem["question"])
        problems.append(problem)
    if len(problems) < num_problems:
        raise RuntimeError(
            f"Generated only {len(problems)} unique Quantum Lock problems after {source_index} official indices."
        )
    return problems, {
        "source_indices_consumed": source_index,
        "duplicates_skipped": source_index - len(problems),
    }
