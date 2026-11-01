from __future__ import annotations


CODE_INSTRUCTION_WITHOUT_STARTER = """You will be given a programming problem. Write a correct Python program that follows the specification and passes all tests.

### Format:
Read input from stdin, solve the problem, and write the answer to stdout.
Do not hard-code the sample inputs.

Please reason step by step before writing code: first state the input
constraints and the time/space complexity your solution must meet, then
outline the algorithm. Keep this reasoning in plain prose. Then output the
complete program as the last thing in your response, inside exactly one
fenced Python code block."""

CODE_INSTRUCTION_WITH_STARTER = """You will be given a programming problem. Write a correct Python program that follows the specification and passes all tests.

### Format:
Use the provided starter code and complete the solution.
Please reason step by step before writing code: first state the input
constraints and the time/space complexity your solution must meet, then
outline the algorithm. Keep this reasoning in plain prose. Then output the
complete program as the last thing in your response, inside exactly one
fenced Python code block."""


def build_user_prompt(problem: str, starter_code: str = "") -> str:
    """Final Code protocol: prose reasoning followed by one Python code block."""
    starter_code = (starter_code or "").strip()
    if starter_code:
        return f"""{CODE_INSTRUCTION_WITH_STARTER}

### Problem:
{problem}

### Starter Code:
<starter_code>
{starter_code}
</starter_code>

### Answer:
"""
    return f"""{CODE_INSTRUCTION_WITHOUT_STARTER}

### Problem:
{problem}

### Answer:
"""


def build_messages(problem: str, starter_code: str = "") -> list[dict[str, str]]:
    return [{"role": "user", "content": build_user_prompt(problem, starter_code)}]
