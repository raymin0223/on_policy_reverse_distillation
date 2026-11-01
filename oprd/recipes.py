"""Experiment matrices, ordered by the sections of arXiv:2609.08798v2.

Only deviations from oprd.paper live here. Checkpoint paths are resolved by
the runner, never embedded in the public recipes.
"""

from dataclasses import dataclass, field
from .paper import TASKS, REASONING_TASKS, ADDITIONAL_TASKS

MAIN = ("grpo", "opd", "kdrl", "oprd")


@dataclass(frozen=True)
class Variant:
    name: str = "default"
    options: tuple[str, ...] = ()
    teacher_step: int | None = None
    reference_step: int | None = None
    hydra: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recipe:
    section: str
    tasks: tuple[str, ...]
    methods: tuple[str, ...] = MAIN
    variants: tuple[Variant, ...] = (Variant(),)
    options: tuple[str, ...] = ()
    hydra: tuple[str, ...] = ()
    teacher_initial: str | None = None
    teacher_group: str | None = None
    teacher_step: int | None = None
    teacher_steps: int | None = None
    method_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)
    kind: str = "rl"


RECIPES = {
    "weak_to_strong_math": Recipe("01_weak_to_strong", ("math",)),
    "weak_to_strong_reasoning": Recipe("01_weak_to_strong", REASONING_TASKS),
    "weak_to_strong_code": Recipe("01_weak_to_strong", ("code",)),
    "weak_to_strong_agent": Recipe("01_weak_to_strong", ("agent",)),
    "multi_teacher": Recipe("02_multi_teacher", ("multi_teacher",)),
    "strong_to_weak_math": Recipe(
        "03_strong_to_weak",
        ("math",),
        options=("--student", "Qwen/Qwen3-1.7B", "--schedule-steps", "45"),
        teacher_initial="Qwen/Qwen3-8B",
        teacher_group="strong_math",
        teacher_step=105,
        method_overrides={"kdrl": (
            "actor_rollout_ref.actor.distillation.kdrl_beta_min=0.001",
            "actor_rollout_ref.actor.distillation.kdrl_delta=0.0001",
        )},
    ),
    "strong_to_weak_knights": Recipe(
        "03_strong_to_weak",
        ("knights_knaves",),
        options=("--student", "Qwen/Qwen3-0.6B", "--schedule-steps", "45"),
        hydra=("actor_rollout_ref.actor.ppo_mini_batch_size=64",),
        teacher_initial="Qwen/Qwen3-8B-Base",
        teacher_group="strong_knights_knaves",
        teacher_step=105,
        method_overrides={"kdrl": (
            "actor_rollout_ref.actor.distillation.kdrl_beta_min=0.001",
            "actor_rollout_ref.actor.distillation.kdrl_delta=0.000088889",
        )},
    ),
    "method_comparison": Recipe(
        "04_method_comparison",
        ("math", "knights_knaves", "string_manipulation"),
        methods=("grpo", "opd", "s2l_po", "opsd", "direct_opd", "w2s_opd", "oprd"),
    ),
    "w2sr_p": Recipe(
        "04_method_comparison",
        ("math", "knights_knaves", "string_manipulation"),
        methods=("w2sr_p",),
        kind="sft",
    ),
    "teacher_strength": Recipe(
        "05_guidance_design",
        ("knights_knaves", "quantum_lock"),
        methods=("oprd",),
        variants=tuple(Variant(f"teacher_{step}", teacher_step=step) for step in (15, 60, 105, 150)),
        teacher_steps=150,
    ),
    "guidance_directions": Recipe(
        "05_guidance_design",
        ("math", "knights_knaves"),
        methods=("oprd",),
        variants=tuple(
            Variant(direction, ("--direction", direction), teacher_step=60)
            for direction in ("teacher_shift", "opd", "opsd")
        ),
    ),
    "guidance_directions_main_teacher": Recipe(
        "05_guidance_design",
        ("math", "knights_knaves"),
        methods=("oprd",),
        variants=tuple(
            Variant(direction, ("--direction", direction)) for direction in ("teacher_shift", "opd", "opsd")
        ),
    ),
    "lambda_strength": Recipe(
        "05_guidance_design",
        ("knights_knaves",),
        methods=("oprd",),
        variants=tuple(
            Variant(f"lambda_{value}", ("--lambda", str(value))) for value in (0.0, 0.05, 0.1, 0.5, 1.0)
        ),
    ),
    "alignment_branches": Recipe(
        "05_guidance_design",
        ("math", "knights_knaves"),
        methods=("oprd",),
        variants=tuple(
            Variant(branch, ("--branch", branch)) for branch in ("positive_only", "negative_only")
        ),
    ),
    "branch_schedules": Recipe(
        "05_guidance_design",
        ("math", "knights_knaves"),
        methods=("oprd",),
        variants=(
            Variant("ramp_negative", ("--branch", "ramp_negative")),
            Variant("decay_positive", ("--branch", "decay_positive")),
            Variant("symmetric", ("--branch", "symmetric", "--lambda", "1.0")),
        ),
    ),
    "math_base": Recipe("06_additional_results", ("math_base",)),
    "additional_reasoning": Recipe("06_additional_results", ADDITIONAL_TASKS),
    "reference_length_bias": Recipe(
        "06_additional_results",
        ("color_cube_rotation", "binary_matrix"),
        methods=("oprd",),
        variants=(Variant("base_reference", reference_step=0), Variant("later_reference")),
    ),
    "vanishing_reward": Recipe(
        "06_additional_results",
        ("knights_knaves",),
        options=("--student", "Qwen/Qwen3-1.7B-Base"),
        teacher_initial="Qwen/Qwen3-8B-Base",
        teacher_group="strong_knights_knaves",
        teacher_step=105,
    ),
}


def required_teacher_checkpoints(task: str, group: str | None = None) -> list[int]:
    """Frozen teachers and references consumed from the same teacher directory."""
    group = group or task
    required = set()
    for recipe in RECIPES.values():
        tasks = set(recipe.tasks)
        if "multi_teacher" in tasks:
            tasks.update(REASONING_TASKS)
        if task not in tasks or (recipe.teacher_group or task) != group:
            continue
        for variant in recipe.variants:
            required.add(variant.teacher_step or recipe.teacher_step or TASKS[task].teacher_step)
            reference_step = variant.reference_step
            if reference_step is None:
                reference_step = {"color_cube_rotation": 30, "binary_matrix": 45}.get(task, 0)
            if reference_step:
                required.add(reference_step)
    return sorted(required)
