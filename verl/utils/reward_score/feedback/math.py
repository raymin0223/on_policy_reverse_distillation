# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import os
import re
import signal
from typing import Optional
from math_verify import parse as mv_parse, verify as mv_verify

FORMAT_PENALTY = False
ANSWER_NORMALIZATION_ENV = "MATH_REWARD_ANSWER_NORMALIZATION"
MATH_VERIFY_TIMEOUT_ENV = "MATH_REWARD_VERIFY_TIMEOUT_SECONDS"
NORMALIZED_TIMEOUT_ENV = "MATH_REWARD_NORMALIZED_TIMEOUT_SECONDS"
CORRECTNESS_FEEDBACK_ENV = "MATH_REWARD_CORRECTNESS_FEEDBACK"
MAX_BOXED_ANSWERS_ENV = "MATH_REWARD_MAX_BOXED_ANSWERS"
ANSWER_NORMALIZATION_MODES = {"strict", "normalized"}
DEFAULT_MATH_VERIFY_TIMEOUT_SECONDS = 5.0
DEFAULT_NORMALIZED_TIMEOUT_SECONDS = 1.0
DEFAULT_MAX_BOXED_ANSWERS = 3


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def _clean_extracted_answer(answer: str) -> str:
    answer = re.sub(r"\n\s*", "", answer).strip()
    if answer.startswith(":"):
        answer = answer[1:].strip()
    return answer.rstrip("./").strip()


def extract_boxed_answer(string: str) -> Optional[str]:
    """Extract the answer from the last boxed expression in the whole response."""
    if "boxed" not in string:
        return None

    boxed = last_boxed_only_string(string)
    if boxed is not None:
        return _clean_extracted_answer(remove_boxed(boxed))

    # DAPO sample scripts accept loose forms such as "\boxed 52" or "boxed 52".
    suffix = string.split("boxed")[-1]
    if not suffix:
        return ""
    suffix = suffix.strip()
    if suffix.startswith("{"):
        stack = 1
        answer = []
        for char in suffix[1:]:
            if char == "{":
                stack += 1
                answer.append(char)
            elif char == "}":
                stack -= 1
                if stack == 0:
                    break
                answer.append(char)
            else:
                answer.append(char)
        return _clean_extracted_answer("".join(answer)) if stack == 0 else ""
    return _clean_extracted_answer(suffix.split("$")[0])


def count_boxed_answers(string: str) -> int:
    return len(re.findall(r"\\+boxed\b", string or ""))


def _max_boxed_answers(extra_info: dict | None = None) -> int:
    raw_value = None
    if isinstance(extra_info, dict):
        raw_value = extra_info.get("max_boxed_answers")
    if raw_value in (None, ""):
        raw_value = os.environ.get(MAX_BOXED_ANSWERS_ENV, DEFAULT_MAX_BOXED_ANSWERS)
    try:
        return max(int(raw_value), 0)
    except (TypeError, ValueError):
        return DEFAULT_MAX_BOXED_ANSWERS


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = r"\boxed{"
    # assert s[: len(left)] == left, f"box error: {s}"
    # assert s[-1] == "}", f"box error: {s}"
    if s and s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


def _strip_outer_math_delimiters(text: str) -> str:
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        return text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]") and len(text) >= 4:
        return text[2:-2].strip()
    if text.startswith(r"\(") and text.endswith(r"\)") and len(text) >= 4:
        return text[2:-2].strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        return text[1:-1].strip()
    return text


def _unwrap_text_commands(text: str) -> str:
    text_command_re = re.compile(r"\\(?:text|mathrm|operatorname|mathbf|mathit)\s*\{([^{}]*)\}")
    previous = None
    while previous != text:
        previous = text
        text = text_command_re.sub(r"\1", text)
    return text


def _canonical_answer_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = _strip_outer_math_delimiters(text)
    text = _unwrap_text_commands(text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\(": "",
        r"\)": "",
        r"\[": "",
        r"\]": "",
        r"\,": "",
        r"\!": "",
        r"\;": "",
        r"\:": "",
        r"\cdot": "*",
        r"\times": "*",
        "×": "*",
        "−": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。")
    return text


def _parse_braced_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
    return None


def _replace_latex_frac(text: str) -> str:
    marker = r"\frac"
    out = []
    idx = 0
    while idx < len(text):
        frac_idx = text.find(marker, idx)
        if frac_idx < 0:
            out.append(text[idx:])
            break
        out.append(text[idx:frac_idx])
        num_start = frac_idx + len(marker)
        num_group = _parse_braced_group(text, num_start)
        if num_group is None:
            out.append(marker)
            idx = num_start
            continue
        num, den_start = num_group
        den_group = _parse_braced_group(text, den_start)
        if den_group is None:
            out.append(marker)
            idx = num_start
            continue
        den, idx = den_group
        out.append(f"(({_replace_latex_frac(num)})/({_replace_latex_frac(den)}))")
    return "".join(out)


def _latexish_to_sympy_text(text: str) -> str:
    text = _strip_outer_math_delimiters(text)
    text = _unwrap_text_commands(text)
    text = _replace_latex_frac(text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\(": "",
        r"\)": "",
        r"\[": "",
        r"\]": "",
        r"\,": " ",
        r"\!": "",
        r"\;": " ",
        r"\:": " ",
        r"\cdot": "*",
        r"\times": "*",
        r"\pi": "pi",
        "^": "**",
        "{": "(",
        "}": ")",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _sympy_equivalent(pred: str, answer: str) -> bool:
    if "=" in pred or "=" in answer:
        return False
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        pred_expr = parse_expr(_latexish_to_sympy_text(pred), transformations=transformations, evaluate=True)
        answer_expr = parse_expr(
            _latexish_to_sympy_text(answer), transformations=transformations, evaluate=True
        )
        return bool(sp.simplify(pred_expr - answer_expr) == 0)
    except Exception:
        return False


def _extract_mc_options(extra_info: dict | None) -> dict[str, str]:
    if not isinstance(extra_info, dict):
        return {}
    problem = extra_info.get("problem") or extra_info.get("description") or ""
    if not isinstance(problem, str):
        return {}

    options: dict[str, str] = {}
    patterns = [
        re.compile(
            r"\(([A-Z])\)\s*(.*?)(?=\s*\([A-Z]\)\s*|(?:\n|\\n)?\s*Please reason\b|$)",
            flags=re.DOTALL,
        ),
        re.compile(
            r"^\s*([A-Z])\.\s*(.*?)(?=^\s*[A-Z]\.\s*|\n\s*\n|\\n\s*\\n|(?:\n|\\n)?\s*Please reason\b|$)",
            flags=re.DOTALL | re.MULTILINE,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(problem):
            label = match.group(1)
            value = match.group(2).strip()
            if value:
                options[label] = value
    return options


def _normalized_answers_match(pred: str, answer: str, extra_info: dict | None = None) -> bool:
    pred_key = _canonical_answer_text(pred)
    answer_key = _canonical_answer_text(answer)
    if not pred_key or not answer_key:
        return False
    if pred_key == answer_key:
        return True

    options = _extract_mc_options(extra_info)
    if options:
        pred_label = pred_key.upper()
        answer_label = answer_key.upper()
        pred_is_explicit_label = pred_key == pred_label
        if (
            pred_is_explicit_label
            and pred_label in options
            and _canonical_answer_text(options[pred_label]) == answer_key
        ):
            return True
        if answer_label in options and _canonical_answer_text(options[answer_label]) == pred_key:
            return True
    elif pred_key.casefold() == answer_key.casefold():
        return True

    return _sympy_equivalent(pred, answer)


def _resolve_answer_normalization(extra_info: dict | None) -> str:
    mode = None
    if isinstance(extra_info, dict):
        mode = extra_info.get("answer_normalization") or extra_info.get("math_answer_normalization")
    if mode is None:
        mode = os.environ.get(ANSWER_NORMALIZATION_ENV, "strict")
    mode = str(mode).strip().lower()
    if mode not in ANSWER_NORMALIZATION_MODES:
        raise ValueError(
            f"answer_normalization must be one of {sorted(ANSWER_NORMALIZATION_MODES)}, got {mode!r}"
        )
    return mode


def _get_timeout_seconds(env_name: str, default: float) -> float:
    raw_value = os.environ.get(env_name)
    if raw_value in (None, ""):
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return max(value, 0.0)


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        self._previous_handler = None

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if self.seconds <= 0:
            return self
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        return self

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._previous_handler is not None:
                signal.signal(signal.SIGALRM, self._previous_handler)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction.
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]

    # Extract and check the final boxed answer from the full response. Long
    # reasoning tails should not hide a valid boxed answer earlier in the text.
    extracted_pred = extract_boxed_answer(pred)

    return extracted_pred == gt, extracted_pred


def verify(
    solution_str: str,
    answer: str,
    pause_tokens_index: Optional[list[int]] = None,
    answer_normalization: str = "strict",
    extra_info: dict | None = None,
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    # try Math-Verify equivalence check
    if not correct and pred != "":
        try:
            with timeout(
                seconds=_get_timeout_seconds(MATH_VERIFY_TIMEOUT_ENV, DEFAULT_MATH_VERIFY_TIMEOUT_SECONDS)
            ):
                gold_expr = mv_parse(answer)
                pred_expr = mv_parse(pred)
                correct = mv_verify(gold_expr, pred_expr)
        except Exception:  # ignore any parsing/verification errors
            pass
    if not correct and pred != "" and answer_normalization == "normalized":
        try:
            with timeout(
                seconds=_get_timeout_seconds(NORMALIZED_TIMEOUT_ENV, DEFAULT_NORMALIZED_TIMEOUT_SECONDS)
            ):
                correct = _normalized_answers_match(pred, answer, extra_info)
        except Exception:
            correct = False
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, 0 for incorrect)
    """
    extra_info = extra_info or {}
    correctness_feedback = correctness_feedback or os.getenv(
        CORRECTNESS_FEEDBACK_ENV, ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)
    answer_normalization = _resolve_answer_normalization(extra_info)
    boxed_answer_count = count_boxed_answers(solution_str)
    too_many_boxed_answers = boxed_answer_count > _max_boxed_answers(extra_info)

    # Verify the solution
    if too_many_boxed_answers:
        correct, pred = False, ""
    else:
        correct, pred = verify(
            solution_str,
            ground_truth,
            pause_tokens_index,
            answer_normalization=answer_normalization,
            extra_info=extra_info,
        )

    incorrect_format = too_many_boxed_answers or pred is None or pred == ""
    reward = 1.0 if correct and not incorrect_format else 0.0
    score = reward
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    # Generate explicit feedback for format errors (analogous to code feedback)
    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        if too_many_boxed_answers:
            feedback = "Your answer had the wrong format. Do not repeat boxed answers; put only the final answer in \\boxed{...}."
        else:
            feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "boxed_answer_count": boxed_answer_count,
        "too_many_boxed_answers": 1 if too_many_boxed_answers else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "answer_normalization": answer_normalization,
        "feedback": feedback,
    }
