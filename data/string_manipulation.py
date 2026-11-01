"""Deterministic local adapter for Reasoning Gym's ``string_manipulation`` task.

Adapted from open-thought/reasoning-gym (Apache-2.0). The indexed RNG,
twenty ordered rewrite rules, stopping condition, prompt, and default
difficulty match the upstream implementation. Exact duplicate questions are
dropped while extending the indexed stream so train and test stay disjoint.
"""

from __future__ import annotations

import random
from typing import Any, Callable


QUESTION_TEMPLATE = """Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
{rules}

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Your output should be the final transformed string after applying all the rules.

Transform the following string according to the above list of rules:
{string}
"""

RuleFn = Callable[[str], tuple[str, int]]


def _rule_01(value: str) -> tuple[str, int]:
    return ("ca" + value[2:], 1) if value.startswith("ab") else (value, 0)


def _rule_02(value: str) -> tuple[str, int]:
    return (value[:-2] + "cb", 2) if value.endswith("ac") else (value, 0)


def _rule_03(value: str) -> tuple[str, int]:
    return (value[2:] + "aa", 3) if value.startswith("bc") else (value, 0)


def _rule_04(value: str) -> tuple[str, int]:
    return (value[:-2], 4) if value.endswith("bb") else (value, 0)


def _rule_05(value: str) -> tuple[str, int]:
    return ("aa" + value[2:-1], 5) if value.startswith("cb") and len(value) > 1 else (value, 0)


def _rule_06(value: str) -> tuple[str, int]:
    return ("bb" + value[2:] + "c", 6) if value.startswith("ca") else (value, 0)


def _rule_07(value: str) -> tuple[str, int]:
    return ("a" + value[:-2] + "b", 7) if value.endswith("cc") else (value, 0)


def _rule_08(value: str) -> tuple[str, int]:
    return (value[1:], 8) if value.startswith("aa") else (value, 0)


def _rule_09(value: str) -> tuple[str, int]:
    return (value.replace("abc", "cab", 1), 9) if "abc" in value else (value, 0)


def _rule_10(value: str) -> tuple[str, int]:
    return (value.replace("bca", "", 1), 10) if "bca" in value else (value, 0)


def _rule_11(value: str) -> tuple[str, int]:
    return (value[:-2] + "ab", 11) if value.endswith("ba") else (value, 0)


def _rule_12(value: str) -> tuple[str, int]:
    return (value[2:], 12) if value.startswith("cc") else (value, 0)


def _rule_13(value: str) -> tuple[str, int]:
    return (value.replace("acb", "bca", 1), 13) if "acb" in value else (value, 0)


def _rule_14(value: str) -> tuple[str, int]:
    return (value[:-1], 14) if value.endswith("ca") and value else (value, 0)


def _rule_15(value: str) -> tuple[str, int]:
    return (value[0] + value[2:], 15) if value.startswith("bb") and len(value) >= 2 else (value, 0)


def _rule_16(value: str) -> tuple[str, int]:
    return (value[:-2] + "cc", 16) if value.endswith("aa") else (value, 0)


def _rule_17(value: str) -> tuple[str, int]:
    index = value.find("ca", 1)
    return (value[:index] + value[index + 2 :], 17) if index != -1 else (value, 0)


def _rule_18(value: str) -> tuple[str, int]:
    count = value.count("b")
    return (value + "ab", 18) if count > 0 and count % 2 == 0 else (value, 0)


def _rule_19(value: str) -> tuple[str, int]:
    middle = len(value) // 2
    return (value[:middle] + value[middle + 1 :], 19) if len(value) > 15 else (value, 0)


def _rule_20(value: str) -> tuple[str, int]:
    return ("zz" + value[2:], 20) if value.startswith("ac") else (value, 0)


RULES: tuple[tuple[str, RuleFn], ...] = (
    ("If the string prefix is 'ab', replace it with 'ca'.", _rule_01),
    ("If the string suffix is 'ac', replace it with 'cb'.", _rule_02),
    ("If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.", _rule_03),
    ("If the string suffix is 'bb', delete the last two characters.", _rule_04),
    ("If the string prefix is 'cb', replace it with 'aa' and delete the last character.", _rule_05),
    ("If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.", _rule_06),
    ("If the string suffix is 'cc', replace it with 'b' and prepend 'a' to the start.", _rule_07),
    ("If the string prefix is 'aa', remove the first character.", _rule_08),
    ("If the string contains 'abc', replace the first occurrence with 'cab'.", _rule_09),
    ("If the string contains 'bca', delete the first occurrence entirely.", _rule_10),
    ("If the string ends with 'ba', replace it with 'ab'.", _rule_11),
    ("If the string starts with 'cc', remove the first two characters.", _rule_12),
    ("If the string contains 'acb', replace the first occurrence with its reverse ('bca').", _rule_13),
    ("If the string ends with 'ca', remove the last character.", _rule_14),
    ("If the string starts with 'bb', remove the second character.", _rule_15),
    ("If the string ends with 'aa', replace it with 'cc'.", _rule_16),
    (
        "If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.",
        _rule_17,
    ),
    (
        "If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.",
        _rule_18,
    ),
    ("If the string length is greater than 15, remove the middle character.", _rule_19),
    ("If the string starts with 'ac', replace the first two characters with 'zz'.", _rule_20),
)


def _apply_first(value: str, selected_rules: list[tuple[str, RuleFn]]) -> tuple[str, int]:
    for _, rule in selected_rules:
        new_value, rule_index = rule(value)
        if rule_index:
            return new_value, rule_index
    return value, 0


def transform_string(value: str, selected_rules: list[tuple[str, RuleFn]]) -> tuple[list[str], bool]:
    """Return accepted states and whether termination was caused by a repeated state."""
    states = [value]
    while True:
        new_value, rule_index = _apply_first(states[-1], selected_rules)
        if not rule_index:
            return states, False
        if new_value in states:
            return states, True
        states.append(new_value)


def generate_string_manipulation_problem(
    source_index: int,
    seed: int,
    min_string_length: int = 5,
    max_string_length: int = 20,
    min_num_rules: int = 3,
    max_num_rules: int = 8,
) -> dict[str, Any]:
    """Generate exactly the upstream item at ``source_index``."""
    rng = random.Random(seed + source_index)
    string_length = rng.randint(min_string_length, max_string_length)
    initial_string = "".join(rng.choice(("a", "b", "c")) for _ in range(string_length))
    num_rules = rng.randint(min_num_rules, max_num_rules)
    selected_rules = rng.sample(RULES, num_rules)
    rules_text = "\n".join(
        f"{index}. {description}" for index, (description, _) in enumerate(selected_rules, 1)
    )
    states, repeated_state = transform_string(initial_string, selected_rules)
    return {
        "question": QUESTION_TEMPLATE.format(rules=rules_text, string=initial_string),
        "answer": states[-1],
        "string": initial_string,
        "states": states,
        "selected_rules": [description for description, _ in selected_rules],
        "string_length": string_length,
        "num_rules": num_rules,
        "num_transforms": len(states) - 1,
        "repeated_state_termination": repeated_state,
        "source_index": source_index,
    }


def generate_unique_string_manipulation_problems(
    num_problems: int,
    seed: int,
    min_string_length: int = 5,
    max_string_length: int = 20,
    min_num_rules: int = 3,
    max_num_rules: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate the indexed upstream stream, dropping exact duplicate prompts."""
    if num_problems <= 0:
        raise ValueError("num_problems must be positive")
    if not 5 <= min_string_length <= max_string_length:
        raise ValueError("string-length bounds must satisfy 5 <= min <= max")
    if not 3 <= min_num_rules <= max_num_rules <= len(RULES):
        raise ValueError("rule-count bounds must satisfy 3 <= min <= max <= 20")

    problems: list[dict[str, Any]] = []
    answers_by_question: dict[str, str] = {}
    source_index = 0
    max_candidates = max(num_problems * 4, num_problems + 1000)
    while len(problems) < num_problems and source_index < max_candidates:
        problem = generate_string_manipulation_problem(
            source_index=source_index,
            seed=seed,
            min_string_length=min_string_length,
            max_string_length=max_string_length,
            min_num_rules=min_num_rules,
            max_num_rules=max_num_rules,
        )
        source_index += 1
        previous = answers_by_question.get(problem["question"])
        if previous is not None:
            if previous != problem["answer"]:
                raise RuntimeError("duplicate string-manipulation question has conflicting answers")
            continue
        answers_by_question[problem["question"]] = problem["answer"]
        problems.append(problem)

    if len(problems) != num_problems:
        raise RuntimeError(f"generated only {len(problems)} unique string-manipulation questions")
    return problems, {
        "candidates_consumed": source_index,
        "duplicates_discarded": source_index - len(problems),
        "distinct_questions": len(problems),
    }
