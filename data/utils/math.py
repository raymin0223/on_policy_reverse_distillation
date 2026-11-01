import re


def _first_nonempty_text(*values) -> str:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def _extract_assistant_reply(records, speaker_key: str, content_key: str) -> str:
    if not isinstance(records, list):
        return ""

    for record in records:
        if not isinstance(record, dict):
            continue
        speaker = _first_nonempty_text(record.get(speaker_key)).lower()
        if speaker == "assistant":
            return _first_nonempty_text(record.get(content_key))
    return ""


def _normalize_openthoughts_problem(problem: str) -> str:
    problem = _first_nonempty_text(problem)
    if not problem:
        return ""

    # Some rows repeat the training instruction in the raw prompt fields.
    problem = re.sub(r"^Return your final response within\s*\\+boxed\{\}\.\s*", "", problem)
    return problem.strip()


def _extract_braced_content(text: str, start_idx: int) -> str:
    depth = 0
    content_start = start_idx + 1

    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
            if depth == 1:
                content_start = idx + 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:idx]

    return ""


def extract_boxed_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    for marker in (r"\boxed{", r"\fbox{"):
        marker_idx = text.rfind(marker)
        if marker_idx >= 0:
            brace_idx = text.find("{", marker_idx)
            if brace_idx >= 0:
                answer = _extract_braced_content(text, brace_idx).strip()
                if answer:
                    return answer

    hash_match = re.search(r"####\s*([^\n]+)", text)
    if hash_match:
        return hash_match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def process_gsm8k(ex):
    full_solution = ex["answer"]
    # Keep this aligned with verl.utils.reward_score.gsm8k: GSM8K answers can
    # be negative and may include thousands separators.
    m = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", full_solution)
    ans_str = m.group(1) if m else ""
    return {
        "solution": full_solution,
        "answer": ans_str,
    }


def process_competition_math(ex):
    solution = ex["solution"]
    return {
        "solution": solution,
        "answer": extract_boxed_answer(solution),
    }


def process_openthoughts_math(ex):
    problem = _normalize_openthoughts_problem(_first_nonempty_text(ex.get("problem"), ex.get("Question")))
    solution = _first_nonempty_text(ex.get("solution"), ex.get("COT_Reason"))
    if not solution:
        solution = _extract_assistant_reply(ex.get("messages"), speaker_key="role", content_key="content")
    if not solution:
        solution = _extract_assistant_reply(ex.get("conversations"), speaker_key="from", content_key="value")

    answer_source = _first_nonempty_text(ex.get("Answer"))
    answer = extract_boxed_answer(answer_source) if answer_source else ""
    if not answer:
        answer = extract_boxed_answer(solution)

    return {
        "problem": problem,
        "solution": solution,
        "answer": answer,
    }
