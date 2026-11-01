from __future__ import annotations
from fractions import Fraction
import ast
import json
import math
import re
from typing import Any
import sympy

VALID_ROLES = {
    "knight",
    "knave",
    "pioneer",
    "laggard",
    "saint",
    "sinner",
    "hero",
    "villain",
    "angel",
    "devil",
    "altruist",
    "egoist",
    "sage",
    "fool",
}
LATEX_TEXT_COMMANDS = "text|textrm|textbf|textit|mathrm|operatorname"
COLOR_CUBE_FILLER_WORDS = {"a", "an", "colour", "color", "cube", "face", "is", "now", "of", "side", "the"}


def last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
    return text[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(boxed: str) -> str:
    prefix = "\\boxed{"
    if boxed.startswith(prefix) and boxed.endswith("}"):
        return boxed[len(prefix) : -1].strip()
    return ""


def extract_boxed_answer(text: str) -> tuple[str | None, bool]:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return (None, False)
    return (remove_boxed(boxed), True)


def normalize_knights_answer(answer: str) -> set[tuple[str, str]]:
    normalized = answer.lower()
    normalized = re.sub("[\\.,;:\\(\\)\\[\\]\\{\\}\\\"']", " ", normalized)
    parts = normalized.replace(" and ", " ").split()
    assignments = set()
    current_name = None
    for part in parts:
        if part in {"is", "are", "a", "an", "the", "both"}:
            continue
        if part in VALID_ROLES:
            if current_name:
                assignments.add((current_name, part))
                current_name = None
        else:
            current_name = part
    return assignments


def score_knights_knaves(prediction: str, ground_truth: str) -> tuple[bool, str]:
    oracle_assignments = normalize_knights_answer(str(ground_truth))
    answer_assignments = normalize_knights_answer(prediction)
    return (oracle_assignments == answer_assignments, prediction)


def _coerce_int_matrix(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list) or not value:
        return None
    matrix = []
    for row in value:
        if not isinstance(row, list) or not row:
            return None
        coerced_row = []
        for item in row:
            if isinstance(item, bool):
                return None
            try:
                coerced_row.append(int(item))
            except (TypeError, ValueError):
                return None
        matrix.append(coerced_row)
    width = len(matrix[0])
    if any((len(row) != width for row in matrix)):
        return None
    return matrix


def parse_matrix_answer(answer: str, expected_shape: tuple[int, int] | None = None) -> list[list[int]] | None:
    stripped = answer.strip()
    if not stripped:
        return None
    try:
        parsed = ast.literal_eval(stripped)
        matrix = _coerce_int_matrix(parsed)
        if matrix is not None:
            return matrix
    except (SyntaxError, ValueError, TypeError):
        pass
    lines = [line for line in stripped.splitlines() if line.strip()]
    row_values = [[int(x) for x in re.findall("-?\\d+", line)] for line in lines]
    row_values = [row for row in row_values if row]
    if row_values and len({len(row) for row in row_values}) == 1:
        if expected_shape is None or (len(row_values), len(row_values[0])) == expected_shape:
            return row_values
    flat_values = [int(x) for x in re.findall("-?\\d+", stripped)]
    if expected_shape is not None:
        rows, cols = expected_shape
        if rows > 0 and cols > 0 and (len(flat_values) == rows * cols):
            return [flat_values[i * cols : (i + 1) * cols] for i in range(rows)]
    return None


def score_binary_matrix(prediction: str, ground_truth: str) -> tuple[bool, str]:
    expected = parse_matrix_answer(str(ground_truth))
    if expected is None:
        return (False, prediction)
    expected_shape = (len(expected), len(expected[0]))
    predicted = parse_matrix_answer(prediction, expected_shape=expected_shape)
    return (predicted == expected, "" if predicted is None else matrix_to_str(predicted))


def matrix_to_str(matrix: list[list[int]]) -> str:
    return "\n".join((" ".join((str(x) for x in row)) for row in matrix))


def parse_countdown_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _safe_eval_arithmetic(expr: str) -> "Fraction | None":
    """Evaluate an integer arithmetic expression (+, -, *, /, parentheses, unary +/-).

    Uses exact rational arithmetic and rejects anything outside the allowed grammar.
    Returns None when the expression is malformed or divides by zero.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div)

    def _eval(n: ast.AST) -> "Fraction":
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp) and isinstance(n.op, allowed_binops):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            operand = _eval(n.operand)
            return operand if isinstance(n.op, ast.UAdd) else -operand
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and (not isinstance(n.value, bool)):
            return Fraction(n.value)
        if isinstance(n, getattr(ast, "Num", ())) and isinstance(getattr(n, "n", None), int):
            return Fraction(n.n)
        raise ValueError("Disallowed token in arithmetic expression")

    try:
        return _eval(node)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def score_countdown(prediction: str, ground_truth: Any) -> tuple[bool, str]:
    payload = parse_countdown_ground_truth(ground_truth)
    expected_numbers = payload.get("numbers")
    target = payload.get("target")
    if not isinstance(expected_numbers, list) or target is None:
        return (False, prediction)
    expr = prediction.strip()
    if not expr or "=" in expr:
        return (False, prediction)
    used_numbers = [int(m) for m in re.findall("\\d+", expr)]
    if sorted(used_numbers) != sorted((int(x) for x in expected_numbers)):
        return (False, expr)
    value = _safe_eval_arithmetic(expr)
    if value is None:
        return (False, expr)
    return (value == int(target), expr)


def parse_zebra_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"answer": str(ground_truth), "answer_type": "attribute"}


def _normalize_zebra_text(text: str) -> str:
    normalized = text.strip().lower().rstrip(".")
    normalized = re.sub("[\\\"'`]", "", normalized)
    normalized = re.sub("\\s+", " ", normalized)
    return normalized.strip()


def score_zebra_puzzles(prediction: str, ground_truth: Any) -> tuple[bool, str]:
    payload = parse_zebra_ground_truth(ground_truth)
    expected = str(payload.get("answer", ""))
    answer_type = payload.get("answer_type", "attribute")
    pred = prediction.strip()
    if answer_type == "position":
        match = re.search("-?\\d+", pred)
        if match is None:
            return (False, pred)
        try:
            value = int(match.group())
            return (value == int(expected), str(value))
        except ValueError:
            return (False, pred)
    normalized_pred = _normalize_zebra_text(pred)
    normalized_expected = _normalize_zebra_text(expected)
    if not normalized_expected:
        return (False, pred)
    correct = normalized_pred == normalized_expected or normalized_expected in normalized_pred.split()
    return (correct, pred)


def normalize_color_cube_answer(answer: str) -> list[str]:
    """Reduce a cube-face answer to its content words (colors), dropping filler.

    ``\\text{red}``, ``the red side`` and ``Red.`` all normalize to ``["red"]``.
    """
    normalized = answer.strip().lower()
    normalized = re.sub(f"\\\\(?:{LATEX_TEXT_COMMANDS})\\s*\\{{([^{{}}]*)\\}}", "\\1", normalized)
    normalized = re.sub("[^a-z\\s]", " ", normalized)
    return [token for token in normalized.split() if token not in COLOR_CUBE_FILLER_WORDS]


def score_color_cube_rotation(prediction: str, ground_truth: str) -> tuple[bool, str]:
    expected = normalize_color_cube_answer(str(ground_truth))
    predicted = normalize_color_cube_answer(prediction)
    normalized_pred = " ".join(predicted)
    if not expected:
        return (False, normalized_pred)
    return (predicted == expected, normalized_pred)


def score_string_manipulation(prediction: str, ground_truth: Any) -> tuple[bool, str]:
    """The upstream task requires an exact, case-sensitive final string."""
    normalized = prediction.strip()
    return (normalized == str(ground_truth).strip(), normalized)


def _normalize_quantum_lock_sequence(answer: str) -> list[str]:
    """Match the official scorer's permissive A/B/C sequence extraction."""
    return [character.upper() for character in re.findall("[A-C]", answer.upper())]


def score_quantum_lock(prediction: str, ground_truth: Any) -> tuple[float, str]:
    """Score exact, equal-length valid, and longer valid Quantum Lock paths."""
    try:
        oracle = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if not isinstance(oracle, dict):
            return (0.0, prediction)
        predicted_path = _normalize_quantum_lock_sequence(prediction)
        target_path = _normalize_quantum_lock_sequence(str(oracle["answer"]))
        normalized = " → ".join(predicted_path)
        if predicted_path == target_path:
            return (1.0, normalized)
        value = int(oracle["initial_value"])
        color = str(oracle["initial_state"])
        buttons = {str(button["name"]): button for button in oracle["buttons"]}
        for name in predicted_path:
            button = buttons.get(name)
            if button is None or button["active_state"] not in (color, "any"):
                continue
            if button["type"] == "add":
                value += int(button["value"])
            elif button["type"] == "subtract":
                value -= int(button["value"])
            elif button["type"] == "multiply":
                value *= int(button["value"])
            else:
                return (0.0, normalized)
            color = "green" if color == "red" else "red"
        if value == int(oracle["target_value"]):
            return (1.0 if len(predicted_path) == len(target_path) else 0.5, normalized)
        return (0.0, normalized)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return (0.0, prediction)


SCORERS = {
    "knights_knaves": score_knights_knaves,
    "countdown": score_countdown,
    "zebra_puzzles": score_zebra_puzzles,
    "color_cube_rotation": score_color_cube_rotation,
    "binary_matrix": score_binary_matrix,
    "quantum_lock": score_quantum_lock,
    "string_manipulation": score_string_manipulation,
}


def compute_score(solution_str, ground_truth, extra_info=None, task=None):
    if task not in SCORERS:
        raise ValueError(f"Unsupported paper reasoning task: {task}")
    answer, valid = extract_boxed_answer(solution_str)
    if not valid or not answer:
        return {"score": 0.0, "acc": 0.0, "incorrect_format": 1.0}
    score, feedback = SCORERS[task](answer, ground_truth)
    return {"score": float(score), "acc": float(score), "incorrect_format": 0.0, "feedback": feedback}
