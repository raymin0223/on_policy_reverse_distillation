"""Run section-level paper recipes; --dry-run prints a plan without side effects."""

import argparse
from copy import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .paper import TASKS, REASONING_TASKS, MATH_BENCHMARKS, CODE_BENCHMARKS, checkpoint_steps
from .recipes import RECIPES, Variant, required_teacher_checkpoints
from .runtime import tracking

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Step:
    label: str
    command: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    json_path: str | None = None
    json_value: dict | None = None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", choices=RECIPES)
    p.add_argument(
        "--stage",
        choices=(
            "prepare",
            "teacher-sft",
            "teacher",
            "train",
            "evaluate",
            "evaluate-teacher",
            "all",
        ),
        default="train",
    )
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--teacher-root", type=Path, help="Defaults to OUTPUT_ROOT/teachers")
    p.add_argument("--teacher", type=Path, help="Explicit local post-RL checkpoint for a single-task recipe")
    p.add_argument("--teacher-reference", type=Path, help="Explicit local reference checkpoint")
    p.add_argument(
        "--teacher-initialization",
        type=Path,
        help="Existing Agent 3B SFT checkpoint; reuse it instead of teacher-sft in --stage all",
    )
    p.add_argument("--model", type=Path, help="Explicit checkpoint for W2SR-P evaluation")
    p.add_argument("--seed", type=int, help="Defaults to each task's original seed")
    p.add_argument(
        "--teacher-seed", type=int, default=42, help="Shared teacher seed, independent of the student seed"
    )
    p.add_argument(
        "--gpus", type=int, choices=(4,), default=4, help="Paper release: one node with four NVIDIA B200 GPUs"
    )
    p.add_argument("--steps", type=int, nargs="+", help="Evaluation checkpoint selection only")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true", help="Print the plan as JSON; does not execute")
    tracking.add_arguments(p)
    return p


def select(values, requested, kind):
    if requested is None:
        return list(values)
    if len(set(requested)) != len(requested) or set(requested) - set(values):
        raise ValueError(f"Choose unique {kind} from {list(values)}")
    return [v for v in values if v in requested]


def command(module, *args):
    return [sys.executable, "-m", module, *map(str, args)]


def checkpoint(directory, step):
    return directory / f"global_step_{step}" / "actor/huggingface"


def teacher_directory(a, recipe, task):
    return a.teacher_root / (recipe.teacher_group or task) / f"seed-{a.teacher_seed}"


def teacher_checkpoint(a, recipe, task, variant):
    step = variant.teacher_step or recipe.teacher_step or TASKS[task].teacher_step
    return a.teacher or checkpoint(teacher_directory(a, recipe, task), step)


def reference(a, recipe, task, variant):
    if a.teacher_reference:
        return str(a.teacher_reference)
    if task == "agent":
        return str(a.teacher_initialization or a.data_root / "agent/models/teacher3b_sft")
    step = variant.reference_step
    if step is None:
        step = {"color_cube_rotation": 30, "binary_matrix": 45}.get(task, 0)
    if step:
        return str(checkpoint(teacher_directory(a, recipe, task), step))
    return recipe.teacher_initial or TASKS[task].teacher_base


def run_directory(a, recipe, task, method, variant):
    return a.output_root / recipe.section / a.recipe / task / method / variant.name / f"seed-{a.seed}"


def training_data(a, task):
    if task == "agent":
        return [str(a.data_root / "agent/prepared/train.parquet")]
    tasks = REASONING_TASKS if task == "multi_teacher" else ("math" if task == "math_base" else task,)
    return [str(a.data_root / name / "train.parquet") for name in tasks]


def benchmarks(a, task):
    if task.startswith("math"):
        if a.recipe in {"strong_to_weak_math", "method_comparison", "w2sr_p"}:
            return ("aime24",)
        return MATH_BENCHMARKS
    if task == "multi_teacher":
        return REASONING_TASKS
    if task == "code":
        return CODE_BENCHMARKS
    return (task,)


def evaluate(a, task, model, output, benchmark):
    if task == "agent":
        data = a.data_root / "agent/prepared/holdout_full.parquet"
        cmd = command(
            "oprd.agent.evaluate",
            "--model",
            model,
            "--data",
            data,
            "--seed",
            a.seed,
            "--tensor-parallel-size",
            1,
            "--data-parallel-size",
            a.gpus,
            "--output",
            output,
        )
    else:
        data = a.data_root / benchmark / "test.parquet"
        cmd = command(
            "oprd.evaluate",
            "--task",
            task,
            "--model",
            model,
            "--data",
            data,
            "--seed",
            a.seed,
            "--tensor-parallel-size",
            1,
            "--data-parallel-size",
            a.gpus,
            "--output",
            output,
        )
    return Step(f"Evaluate {benchmark}: {model}", cmd, [str(model), str(data)])


def prepare_steps(a, tasks):
    result = []
    groups = list(
        dict.fromkeys(
            "math" if t.startswith("math") else t if t in {"code", "agent"} else "reasoning" for t in tasks
        )
    )
    for group in groups:
        if group == "agent":
            for part in ("models", "rl", "eval"):
                result.append(
                    Step(
                        f"Agent download {part}",
                        command("oprd.agent.download", "--root", a.data_root / "agent", "--part", part),
                    )
                )
            result.append(
                Step(
                    "Agent prepare RL",
                    command("oprd.agent.prepare", "--root", a.data_root / "agent", "--part", "rl"),
                )
            )
            result.append(
                Step("Agent full holdout", command("oprd.agent.holdout", "--root", a.data_root / "agent"))
            )
        else:
            result.append(
                Step(
                    f"Prepare {group} (data seed 0)",
                    command("oprd.prepare", "--task", group, "--output-root", a.data_root, "--seed", 0),
                )
            )
    return result


def teacher_steps(a, recipe, tasks):
    result = []
    student_horizons = {}
    for student_task in tasks:
        for task in REASONING_TASKS if student_task == "multi_teacher" else (student_task,):
            student_horizons[task] = max(student_horizons.get(task, 0), TASKS[student_task].steps)
    for task, student_horizon in student_horizons.items():
        # Retain the intermediate references and teacher-quality checkpoints.
        # All selected checkpoints come from the same seeded GRPO trajectory.
        spec = TASKS[task]
        # Recipes sharing one teacher directory must preserve each other's
        # selected teachers/references, including later component studies.
        required_steps = required_teacher_checkpoints(task, recipe.teacher_group)
        total = max(student_horizon, recipe.teacher_steps or 0, *required_steps)
        saved_steps = sorted(set(checkpoint_steps(total)) | set(required_steps))
        out = teacher_directory(a, recipe, task)
        student = recipe.teacher_initial or spec.teacher_base
        if task == "agent":
            student = str(a.teacher_initialization or a.data_root / "agent/models/teacher3b_sft")
        cmd = command(
            "oprd.train",
            "--task",
            task,
            "--teacher-training",
            "--student",
            student,
            "--steps",
            total,
            "--seed",
            a.teacher_seed,
            "--gpus",
            a.gpus,
            "--data-root",
            a.data_root,
            "--output",
            out,
            *tracking.arguments(a),
            "trainer.save_steps=" + json.dumps(saved_steps, separators=(",", ":")),
            f"trainer.experiment_name=teacher-{recipe.teacher_group or task}-seed{a.teacher_seed}",
            # recipe.hydra contains student overrides (e.g. the 0.6B minibatch).
            # Teachers retain their task's own training settings.
        )
        result.append(
            Step(
                f"Train {task} teacher ({total} updates)",
                cmd,
                training_data(a, task) + ([student] if task == "agent" else []),
            )
        )
    return result


def train_step(a, recipe, task, method, variant, teachers_file=None):
    out = run_directory(a, recipe, task, method, variant)
    cmd = command(
        "oprd.train",
        "--task",
        task,
        "--method",
        method,
        "--seed",
        a.seed,
        "--gpus",
        a.gpus,
        "--data-root",
        a.data_root,
        "--output",
        out,
        *tracking.arguments(a),
    )
    requires = training_data(a, task)
    if task == "agent":
        student = a.data_root / "agent/models/student7b"
        cmd += ["--student", str(student)]
        requires.append(str(student))
    if method != "grpo":
        if task == "multi_teacher":
            cmd += ["--teachers", str(teachers_file)]
            requires += [str(teacher_checkpoint(a, recipe, t, variant)) for t in REASONING_TASKS]
        else:
            teacher = teacher_checkpoint(a, recipe, task, variant)
            cmd += ["--teacher", str(teacher)]
            requires.append(str(teacher))
        ref = reference(a, recipe, task, variant)
        cmd += ["--teacher-reference", ref]
        if Path(ref).is_absolute():
            requires.append(ref)
    cmd += list(recipe.options) + list(variant.options)
    overrides = list(recipe.hydra) + list(recipe.method_overrides.get(method, ())) + list(variant.hydra)
    overrides.append(f"trainer.experiment_name={a.recipe}-{task}-{method}-{variant.name}-seed{a.seed}")
    cmd += overrides
    return Step(f"Train {task}/{method}/{variant.name}", cmd, requires)


def sft_steps(a, recipe, tasks, stage):
    result = []
    for task in tasks:
        spec = TASKS[task]
        out = run_directory(a, recipe, task, "w2sr_p", Variant())
        data = out / "demonstrations"
        if stage == "train":
            teacher = teacher_checkpoint(a, recipe, task, Variant())
            cmd = command(
                "oprd.build_teacher_data",
                "--task",
                "math_dapo" if task == "math" else task,
                "--data",
                training_data(a, task)[0],
                "--model",
                teacher,
                "--output-dir",
                data,
                "--max-new-tokens",
                spec.response_length,
                "--max-model-len",
                2048 + spec.response_length,
                "--data-seed",
                a.seed,
                "--generation-seed",
                a.seed,
                "--selection-seed",
                a.seed,
                "--data-parallel-size",
                a.gpus,
            )
            result.append(
                Step(
                    f"Build {task} verified teacher demonstrations",
                    cmd,
                    [str(teacher), *training_data(a, task)],
                )
            )
            result.append(
                Step(
                    f"W2SR-P {task} SFT (3 epochs)",
                    command(
                        "oprd.sft",
                        "--model",
                        spec.student,
                        "--data",
                        data / "train.parquet",
                        "--output",
                        out / "sft",
                        "--seed",
                        a.seed,
                        "--gpus",
                        a.gpus,
                        "--max-length",
                        2048 + spec.response_length,
                        *tracking.arguments(a),
                    ),
                    [str(data / "train.parquet")],
                )
            )
        elif stage == "evaluate":
            if not a.model or len(tasks) != 1:
                raise ValueError(
                    "W2SR-P evaluation requires --tasks TASK --model /path/to/exported-sft-checkpoint"
                )
            for benchmark in benchmarks(a, task):
                result.append(
                    evaluate(a, task, a.model, out / "evaluation" / f"{benchmark}.jsonl", benchmark)
                )
    return result


def build_plan(a):
    recipe = RECIPES[a.recipe]
    a.data_root = a.data_root.resolve()
    a.output_root = a.output_root.resolve()
    a.teacher_root = (a.teacher_root or a.output_root / "teachers").resolve()
    for key in ("teacher", "teacher_reference", "model"):
        if getattr(a, key):
            setattr(a, key, getattr(a, key).resolve())
    tasks = select(recipe.tasks, a.tasks, "tasks")
    if a.stage == "prepare":
        return prepare_steps(a, tasks)
    if a.seed is None:
        if a.teacher and len(tasks) > 1:
            raise ValueError("--teacher requires a single task; use --teacher-root for a matrix")
        plans = []
        groups = {}
        for task in tasks:
            groups.setdefault(TASKS[task].seed, []).append(task)
        for seed, selected_tasks in groups.items():
            task_args = copy(a)
            task_args.tasks = selected_tasks
            task_args.seed = seed
            plans.extend(build_plan(task_args))
        return plans
    methods = select(recipe.methods, a.methods, "methods")
    names = select([v.name for v in recipe.variants], a.variants, "variants")
    variants = [v for v in recipe.variants if v.name in names]
    if a.gpus < 1 or a.seed < 0 or ("multi_teacher" in tasks and a.gpus != 4):
        raise ValueError("Use positive GPU count and nonnegative seed; multi-teacher requires four GPUs")
    if a.steps and (
        any(s < 1 or s > TASKS[t].steps for s in a.steps for t in tasks) or len(set(a.steps)) != len(a.steps)
    ):
        raise ValueError("Evaluation steps must be unique and within the training horizon")
    if a.teacher and (len(tasks) != 1 or "multi_teacher" in tasks or a.recipe == "teacher_strength"):
        raise ValueError(
            "--teacher cannot replace a multi-task or teacher-quality checkpoint matrix; use --teacher-root"
        )
    if a.teacher_reference and a.recipe == "reference_length_bias":
        raise ValueError("--teacher-reference would erase the reference ablation; use --teacher-root")
    if a.stage == "teacher-sft" and tasks != ["agent"]:
        raise ValueError("teacher-sft applies only to the Agent recipe")
    if recipe.kind == "sft" and a.stage == "all":
        raise ValueError("W2SR-P uses prepare, teacher, train, then evaluate with the exported --model")
    if a.teacher_initialization and (tasks != ["agent"] or a.stage == "teacher-sft"):
        raise ValueError(
            "--teacher-initialization reuses an Agent SFT checkpoint; it cannot be combined with teacher-sft"
        )
    stages = (
        (
            ["prepare"]
            + (["teacher-sft"] if tasks == ["agent"] and not a.teacher_initialization else [])
            + ["teacher", "train", "evaluate"]
        )
        if a.stage == "all"
        else [a.stage]
    )
    result = []
    for stage in stages:
        if stage == "prepare":
            result += prepare_steps(a, tasks)
        elif stage == "teacher-sft":
            for module in ("download", "prepare"):
                result.append(
                    Step(
                        "Agent SFT data " + module,
                        command(
                            "oprd.agent." + module,
                            "--root",
                            a.data_root / "agent",
                            "--part",
                            "sft",
                            "--exclude-corrupt-sft-sample",
                        ),
                    )
                )
            python = ROOT / ".venv-vilasr-sft/bin/python"
            for mode in ("prepare", "train"):
                result.append(
                    Step(
                        f"Agent SFT {mode}",
                        [
                            str(python),
                            "-m",
                            "oprd.agent.sft",
                            "--root",
                            str(a.data_root / "agent"),
                            "--seed",
                            str(a.teacher_seed),
                            "--mode",
                            mode,
                            *tracking.arguments(a),
                        ],
                        [str(python)],
                    )
                )
        elif stage == "teacher":
            result += teacher_steps(a, recipe, tasks)
        elif recipe.kind == "sft" and stage in {"train", "evaluate"}:
            result += sft_steps(a, recipe, tasks, stage)
        else:
            for task in tasks:
                for variant in variants:
                    if stage == "evaluate-teacher":
                        teacher = teacher_checkpoint(a, recipe, task, variant)
                        if task == "multi_teacher":
                            for domain in REASONING_TASKS:
                                model = teacher_checkpoint(a, recipe, domain, variant)
                                result.append(
                                    evaluate(
                                        a,
                                        domain,
                                        model,
                                        a.output_root
                                        / "teacher_evaluation"
                                        / domain
                                        / f"seed-{a.seed}.jsonl",
                                        domain,
                                    )
                                )
                        else:
                            for b in benchmarks(a, task):
                                result.append(
                                    evaluate(
                                        a,
                                        task,
                                        teacher,
                                        a.output_root
                                        / "teacher_evaluation"
                                        / a.recipe
                                        / task
                                        / variant.name
                                        / f"{b}-seed{a.seed}.jsonl",
                                        b,
                                    )
                                )
                        continue
                    teachers_file = a.output_root / recipe.section / f"teachers-seed{a.seed}.json"
                    if task == "multi_teacher" and stage == "train" and any(m != "grpo" for m in methods):
                        result.append(
                            Step(
                                "Write public local teacher mapping",
                                json_path=str(teachers_file),
                                json_value={
                                    t: str(teacher_checkpoint(a, recipe, t, variant)) for t in REASONING_TASKS
                                },
                            )
                        )
                    for method in methods:
                        if stage == "train":
                            result.append(train_step(a, recipe, task, method, variant, teachers_file))
                        elif stage == "evaluate":
                            spec = TASKS[task]
                            steps = a.steps or checkpoint_steps(spec.steps)
                            out = run_directory(a, recipe, task, method, variant)
                            for step in steps:
                                for b in benchmarks(a, task):
                                    result.append(
                                        evaluate(
                                            a,
                                            task,
                                            checkpoint(out, step),
                                            out / "evaluation" / f"step-{step}-{b}.jsonl",
                                            b,
                                        )
                                    )
    if not result:
        raise ValueError("The selected recipe/stage produced no work")
    return result


def execute(plan):
    for step in plan:
        for name in step.requires:
            path = Path(name)
            if not path.exists():
                raise FileNotFoundError(
                    f"{step.label}: missing {path}. Run the preparation/teacher stage, or select the correct roots/checkpoint."
                )
            if path.name == "huggingface" and not (path / "config.json").is_file():
                raise ValueError(
                    f"{path} is not an exported Hugging Face checkpoint; export/merge the actor first"
                )
        print(f"\n{step.label}", flush=True)
        if step.json_path:
            path = Path(step.json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = json.dumps(step.json_value, indent=2) + "\n"
            if path.exists() and path.read_text() != value:
                raise ValueError(f"Existing teacher mapping differs: {path}")
            path.write_text(value)
        else:
            print(shlex.join(step.command), flush=True)
            subprocess.run(step.command, cwd=ROOT, check=True)


def main():
    p = parser()
    a = p.parse_args()
    try:
        plan = build_plan(a)
        if a.json:
            print(json.dumps([asdict(step) for step in plan], indent=2))
        elif a.dry_run:
            print(f"W&B: mode={a.wandb_mode}, project={a.wandb_project}")
            for step in plan:
                print(f"\n# {step.label}")
                print(
                    shlex.join(step.command)
                    if step.command
                    else json.dumps({step.json_path: step.json_value}, indent=2)
                )
        else:
            tracking.configure_environment(a)
            execute(plan)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
