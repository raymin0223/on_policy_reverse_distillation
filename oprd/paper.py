"""Experiment definitions from arXiv:2609.08798v2, including its appendix."""

from dataclasses import dataclass

METHODS = ("grpo", "opd", "kdrl", "oprd", "opsd", "direct_opd", "w2s_opd", "s2l_po")
REASONING_TASKS = ("knights_knaves", "quantum_lock", "string_manipulation", "countdown")
ADDITIONAL_TASKS = ("zebra_puzzles", "color_cube_rotation", "binary_matrix")


def checkpoint_steps(total_steps: int) -> list[int]:
    """Five equally spaced model snapshots, including the final update."""
    if total_steps < 1:
        raise ValueError("Training horizon must be positive")
    return sorted({(total_steps * fraction + 4) // 5 for fraction in range(1, 6)})


@dataclass(frozen=True)
class Task:
    student: str
    teacher_base: str
    teacher_step: int
    steps: int = 150
    mini_batch: int = 32
    response_length: int = 8192
    evaluation_length: int = 8192
    evaluation_n: int = 1
    evaluation_temperature: float = 0.6
    evaluation_top_p: float = 0.95
    schedule_steps: int = 75
    validation_interval: int = 15
    system_prompt: bool = False
    length_penalty_buffer: int = 0
    seed: int = 42
    oprd_lambda: float = 0.5


TASKS = {
    "math": Task(
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B",
        75,
        mini_batch=64,
        response_length=20480,
        evaluation_length=38912,
        evaluation_n=16,
        evaluation_temperature=0.7,
        evaluation_top_p=0.8,
        schedule_steps=30,
        system_prompt=True,
        length_penalty_buffer=4096,
    ),
    "math_base": Task(
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-4B-Base",
        90,
        mini_batch=32,
        response_length=20480,
        evaluation_length=30720,
        evaluation_n=16,
        evaluation_temperature=0.7,
        evaluation_top_p=0.8,
        schedule_steps=75,
        system_prompt=False,
        length_penalty_buffer=4096,
    ),
    **{
        name: Task(
            "Qwen/Qwen3-8B-Base",
            "Qwen/Qwen3-4B-Base",
            75 if name == "string_manipulation" else 105,
            seed=1234 if name == "binary_matrix" else 42,
            oprd_lambda=1.0 if name == "binary_matrix" else 0.5,
        )
        for name in REASONING_TASKS + ADDITIONAL_TASKS
    },
    "code": Task(
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen3-8B-Base",
        435,
        response_length=16384,
        evaluation_length=30720,
        evaluation_n=4,
        evaluation_temperature=0.7,
        evaluation_top_p=0.8,
        schedule_steps=45,
        length_penalty_buffer=4096,
    ),
    "multi_teacher": Task(
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-4B-Base",
        105,
        steps=300,
    ),
    "agent": Task(
        "ViLaSR-SFT-7B",
        "ViLaSR-SFT-3B",
        450,
        steps=225,
        mini_batch=64,
        response_length=16384,
        evaluation_length=16384,
        schedule_steps=45,
        system_prompt=True,
        oprd_lambda=1.0,
    ),
}
MATH_BENCHMARKS = ("aime24", "aime25", "hmmt25", "olympiadbench")
CODE_BENCHMARKS = ("lcbv5", "humanevalplus", "codecontests", "apps_competition")
AGENT_FAMILIES = ("maze", "gpt4scene", "sr91k", "spatial_vqa")


def mean_score(scores):
    """Mean@m averages all response scores; it is not any-of-m success."""
    values = list(scores)
    if not values:
        raise ValueError("Cannot report an empty evaluation")
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("Evaluation scores must lie in [0, 1]")
    return 100.0 * sum(values) / len(values)
