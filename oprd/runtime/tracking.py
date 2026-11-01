"""Public tracking settings. Authentication is managed outside the repository."""

import os

DEFAULT_PROJECT = "on-policy-reverse-distillation"


def add_arguments(parser):
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default=os.environ.get("WANDB_MODE", "offline"),
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))


def configure_environment(args):
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity


def trainer_overrides(args):
    return {
        "trainer.logger": ["console"] if args.wandb_mode == "disabled" else ["console", "wandb"],
        "trainer.project_name": args.wandb_project,
        "trainer.wandb_mode": args.wandb_mode,
        "trainer.wandb_entity": args.wandb_entity,
    }


def arguments(args):
    result = ["--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project]
    if args.wandb_entity:
        result += ["--wandb-entity", args.wandb_entity]
    return result
