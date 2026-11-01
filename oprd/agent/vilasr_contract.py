"""ViLaSR think/draw/observe protocol, with a strict terminal rule verifier.

Prompts/rendering follow AntResearchNLP/ViLaSR (Apache-2.0), pinned by
vilasr_assets.lock.json. Parsing is deliberately fail-closed, not substring matching.
"""

import ast
from collections import OrderedDict
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from PIL import ImageDraw

VERSION = "vilasr-drawing-terminal-v1"
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 4 * 28 * 28
FOLLOWUP = "Continue your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:"
FINAL = "Then, you can not invoke the Object Mapper or Path Tracer tool. Please answer the initial question and structure your response as required:"
_NUMBER = re.compile("[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?")
_ANSWER = re.compile(
    "<think>(?:(?!</?think>|</?answer>).)*</think>\\s*Action:\\s*Answer\\s*<answer>\\s*([^<>]+?)\\s*</answer>\\s*",
    re.S,
)
_DRAW = re.compile(
    "<think>(?:(?!</?think>|</?answer>).)*</think>\\s*Action:\\s*(Object Mapper|Path Tracer)\\s*```json\\s*(.*?)\\s*```\\s*",
    re.S,
)


def clean_segment(text):
    text = text.strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].rstrip()
    return text


def numeric(text):
    text = str(text).strip()
    if not _NUMBER.fullmatch(text):
        raise ValueError("Expected one finite number, without units or trailing text")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Non-finite answer")
    return value


def mra(pred, target):
    error = abs(pred - target) / (abs(target) + 1e-08)
    return sum((error <= (10 - i) / 20 for i in range(10))) / 10


def answer_score(answer, gold):
    kind, target = (gold["type"], str(gold["answer"]).strip())
    if kind == "multiple choice":
        if not re.fullmatch("[A-Z]", target):
            raise ValueError(f"Invalid MCQ gold: {target}")
        return float(bool(re.fullmatch("[A-Za-z]", answer)) and answer.upper() == target)
    if kind in ("regression", "numerical"):
        target_num = numeric(target)
        try:
            pred = numeric(answer)
        except ValueError:
            return 0.0
        if gold.get("numeric_metric", "mra") == "exact2":
            return float(round(pred, 2) == round(target_num, 2) and ("." in answer) == ("." in target))
        return mra(pred, target_num)
    if kind == "vci":
        keys = (
            "move_right",
            "move_left",
            "move_up",
            "move_down",
            "move_forward",
            "move_backward",
            "rotate_right",
            "rotate_left",
            "rotate_up",
            "rotate_down",
        )

        def parse(s):
            result = {}
            for item in s.split(","):
                k, v = (x.strip() for x in item.split(":"))
                if k not in keys or k in result:
                    raise ValueError("Unknown or repeated VCI component")
                result[k] = numeric(v)
            return result

        truth = parse(target)
        try:
            pred = parse(answer)
        except ValueError:
            return 0.0
        return (
            sum(
                (
                    mra(
                        pred.get(keys[i], 0) - pred.get(keys[i + 1], 0),
                        truth.get(keys[i], 0) - truth.get(keys[i + 1], 0),
                    )
                    for i in range(0, 10, 2)
                )
            )
            / 5
        )
    raise ValueError(f"Unsupported gold type {kind}")


def score_response(segments, stop, gold, interactions=None):
    interactions = list(interactions) if interactions is not None else []
    valid_calls = sum((bool(t.get("valid_action")) for t in interactions))
    segs = list(segments) if segments is not None else []
    match = _ANSWER.fullmatch(clean_segment(segs[-1])) if segs else None
    prior_valid = all((_DRAW.fullmatch(clean_segment(s)) for s in segs[:-1]))
    valid = bool(
        match
        and prior_valid
        and (stop == "answer")
        and (len(interactions) == len(segs) - 1)
        and all((t.get("valid_action") and t.get("observation_accepted") for t in interactions))
    )
    answer = match[1].strip() if match else ""
    accuracy = answer_score(answer, gold) if valid else 0.0
    return {
        "score": accuracy,
        "accuracy": accuracy,
        "parsed_answer": answer,
        "incorrect_format": int(not valid),
        "missing_final_answer": int(match is None),
        "budget_exhausted": int(
            stop
            in {
                "response_budget",
                "action_length_limit",
                "image_limit",
                "max_turns_reached",
                "observation_response_budget",
            }
        ),
        "execution_error": int(stop == "tool_error"),
        "tool_error": int(any((not t.get("valid_action") for t in interactions))),
        "valid_tool_count": valid_calls,
        "curiosity_bonus": 0.0,
        "redundancy_penalty": 0.0,
    }


@lru_cache(maxsize=1)
def prompt_constants():
    return json.loads(Path(__file__).with_name("prompts.json").read_text())


class DrawingSession:
    """Trajectory-local copy-on-write drawing; original files/images never change."""

    def __init__(self, images):
        self.images = images
        self.ancestors = list(range(len(images)))
        self.annotations = [OrderedDict() for _ in images]

    def execute(self, text):
        match = _DRAW.fullmatch(clean_segment(text))
        if not match:
            raise ValueError("Expected one complete think/action/JSON drawing turn")
        try:
            commands = json.loads(match[2])
        except json.JSONDecodeError:
            commands = ast.literal_eval(match[2])
        if not isinstance(commands, list) or not 1 <= len(commands) <= 64:
            raise ValueError("Expected 1..64 drawing commands")
        grouped = OrderedDict()
        field = "bbox_2d" if match[1] == "Object Mapper" else "start_point_2d"
        for command in commands:
            if not isinstance(command, dict) or field not in command:
                raise ValueError("Action name and coordinates disagree")
            index = command.get("index")
            if type(index) is not int or not 1 <= index <= len(self.images):
                raise ValueError("Invalid one-based image index")
            if not isinstance(command.get("label"), str) or len(command["label"]) > 256:
                raise ValueError("Invalid drawing label")
            image = self.images[index - 1]
            coords = (
                [command["bbox_2d"]]
                if field == "bbox_2d"
                else [command["start_point_2d"], command.get("end_point_2d")]
            )
            for points in coords:
                if not isinstance(points, list) or len(points) != (4 if field == "bbox_2d" else 2):
                    raise ValueError("Invalid coordinate shape")
                for j, value in enumerate(points):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or (not math.isfinite(value))
                    ):
                        raise ValueError("Non-finite coordinate")
                    if not 0 <= value <= (image.width if j % 2 == 0 else image.height):
                        raise ValueError("Coordinate outside displayed image")
            grouped.setdefault(index - 1, []).append(command)
        if len(self.images) + len(grouped) > 42:
            raise ValueError("image_limit")
        new_images, states, ancestors = ([], [], [])
        for index, commands in grouped.items():
            state = self.annotations[index].copy()
            for command in commands:
                kind = "box" if "bbox_2d" in command else "line"
                state[kind, command["label"]] = command
            ancestor = self.ancestors[index]
            image = self.images[ancestor].copy()
            draw = ImageDraw.Draw(image)
            colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "brown"]
            box_index = 0
            for (kind, label), c in state.items():
                if kind == "box":
                    x1, y1, x2, y2 = c["bbox_2d"]
                    bounds = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    color = colors[box_index % len(colors)]
                    draw.rectangle(bounds, outline=color, width=4)
                    draw.text((bounds[0] + 8, bounds[1] + 6), label, fill=color)
                    box_index += 1
                else:
                    start, end = (c["start_point_2d"], c["end_point_2d"])
                    draw.line((*start, *end), fill="black", width=4)
                    for point, color in ((start, "red"), (end, "blue")):
                        x, y = point
                        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            new_images.append(image)
            states.append(state)
            ancestors.append(ancestor)
        self.annotations.extend(states)
        self.ancestors.extend(ancestors)
        self.images.extend(new_images)
        return new_images
