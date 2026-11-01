# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import hashlib
import json
import math
import os
import random
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import orjson

_WANDB_NAME_LIMIT = 128
_WANDB_NAME_HASH_LENGTH = 12


def _wandb_step_offset() -> int:
    raw_value = os.environ.get("WANDB_STEP_OFFSET", "0").strip()
    try:
        offset = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"WANDB_STEP_OFFSET must be a non-negative integer, got {raw_value!r}") from exc
    if offset < 0:
        raise ValueError(f"WANDB_STEP_OFFSET must be a non-negative integer, got {offset}")
    return offset


def _wandb_explicit_step(step: int) -> int:
    return int(step) + _wandb_step_offset()


def _fit_wandb_run_name(name: str) -> str:
    if len(name) <= _WANDB_NAME_LIMIT:
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_WANDB_NAME_HASH_LENGTH]
    prefix_length = _WANDB_NAME_LIMIT - _WANDB_NAME_HASH_LENGTH - 1
    return f"{name[:prefix_length]}-{digest}"


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: str | list[str] = "console",
        config=None,
        group_name=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn(
                    "`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2
                )
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False
        self._wandb_step_offset = _wandb_step_offset()
        self._defined_metrics: set[tuple[str, str, str | None]] = set()
        # W&B cannot append to an older explicit step after a newer step was
        # logged. Async validation therefore registers its source step as a
        # barrier: later W&B records are buffered until that validation result
        # has been written at the source step. Other backends remain live.
        self._wandb_step_barriers: set[int] = set()
        self._wandb_deferred_logs: list[tuple[int, int, dict[str, Any], bool]] = []
        self._wandb_deferred_log_sequence = 0
        # Backfill mode appends validation-only records to an existing run whose
        # internal W&B history has already advanced past the source steps. W&B
        # cannot insert explicit historical steps, so these records are appended
        # without `step=` and plotted against their training/global_step field.
        self._wandb_global_step_backfill = os.environ.get(
            "WANDB_GLOBAL_STEP_BACKFILL", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings_kwargs = {}
            if config and config["trainer"].get("wandb_proxy", None):
                settings_kwargs["https_proxy"] = config["trainer"]["wandb_proxy"]
            init_timeout = os.environ.get("WANDB_INIT_TIMEOUT")
            if init_timeout:
                settings_kwargs["init_timeout"] = float(init_timeout)
            settings = wandb.Settings(**settings_kwargs) if settings_kwargs else None
            entity = os.environ.get("WANDB_ENTITY", None)
            trainer_config = config.get("trainer", {}) if config else {}
            entity = trainer_config.get("wandb_entity") or entity
            mode = trainer_config.get("wandb_mode") or os.environ.get("WANDB_MODE", "offline")
            run_directory = trainer_config.get("default_local_dir")
            if run_directory:
                Path(run_directory).mkdir(parents=True, exist_ok=True)
            run_id = os.environ.get("WANDB_RUN_ID")
            resume = os.environ.get("WANDB_RESUME")
            resume_from = os.environ.get("WANDB_RESUME_FROM")
            if self._wandb_global_step_backfill and not run_id:
                raise ValueError("WANDB_GLOBAL_STEP_BACKFILL requires WANDB_RUN_ID")
            if resume and resume_from:
                raise ValueError("WANDB_RESUME and WANDB_RESUME_FROM are mutually exclusive")
            wandb_name = os.environ.get("WANDB_NAME") or str(experiment_name)
            wandb_name = _fit_wandb_run_name(wandb_name)
            wandb_init_kwargs = {}
            if run_id:
                wandb_init_kwargs["id"] = run_id
            if resume_from:
                # Rewind history to an explicit run moment.  W&B forbids
                # combining this with ordinary resume behavior.
                wandb_init_kwargs["resume_from"] = resume_from
            elif run_id:
                # If run_id is provided, default to resumable behavior.
                wandb_init_kwargs["resume"] = resume or "allow"
            elif resume:
                wandb_init_kwargs["resume"] = resume

            wandb.init(
                mode=mode,
                dir=run_directory,
                save_code=False,
                project=project_name,
                name=wandb_name,
                entity=entity,
                # Do not replace the original run config with the temporary
                # 150-step backfill schedule.
                config=None if self._wandb_global_step_backfill else config,
                settings=settings,
                group=group_name,
                **wandb_init_kwargs,
            )
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Some cloud providers like Azure ML or Databricks automatically set MLFLOW_RUN_ID
            # If set, attach to the existing run instead of creating a new one
            run_id = os.environ.get("MLFLOW_RUN_ID")
            if run_id:
                mlflow.start_run(run_id=run_id)
            else:
                # Project_name is actually experiment_name in MLFlow
                # If experiment does not exist, will create a new experiment
                experiment = mlflow.set_experiment(project_name)
                mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)

            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None, *, wandb_commit: bool = False):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                self._log_to_backend(
                    default_backend,
                    logger_instance,
                    data=data,
                    step=step,
                    wandb_commit=wandb_commit,
                )

    def log_async_validation(self, data, *, source_step: int, collection_step: int) -> None:
        """Log async validation at its source step on W&B without backfilling other backends."""

        source_step = int(source_step)
        collection_step = max(int(collection_step), source_step)
        for default_backend, logger_instance in self.logger.items():
            log_step = source_step if default_backend == "wandb" else collection_step
            self._log_to_backend(
                default_backend,
                logger_instance,
                data=data,
                step=log_step,
                wandb_commit=True,
                completes_wandb_barrier=default_backend == "wandb",
            )

    def _log_to_backend(
        self,
        backend: str,
        logger_instance,
        *,
        data,
        step: int,
        wandb_commit: bool,
        completes_wandb_barrier: bool = False,
    ) -> None:
        if backend == "wandb" and self._wandb_global_step_backfill:
            # Ordinary per-update records would duplicate the original run. Only
            # completed async validation records are appended in this mode.
            if not completes_wandb_barrier:
                return
            logger_instance.log(data=data, commit=True)
            return
        if backend == "wandb" and self._should_defer_wandb_log(step):
            self._wandb_deferred_logs.append(
                (int(step), self._wandb_deferred_log_sequence, dict(data), bool(wandb_commit))
            )
            self._wandb_deferred_log_sequence += 1
            return
        if backend == "wandb":
            # A normal training record must keep a validation source step open
            # until the async validation record has been appended.  That final
            # validation record is the only call allowed to close the barrier.
            if int(step) in self._wandb_step_barriers and not completes_wandb_barrier:
                wandb_commit = False
            logger_instance.log(
                data=data,
                step=int(step) + self._wandb_step_offset,
                commit=bool(wandb_commit),
            )
            return
        logger_instance.log(data=data, step=step)

    def add_wandb_step_barrier(self, step: int) -> None:
        """Keep W&B from advancing past an unfinished async-validation step."""

        if "wandb" in self.logger and not self._wandb_global_step_backfill:
            self._wandb_step_barriers.add(int(step))

    def release_wandb_step_barrier(self, step: int) -> None:
        """Release one async-validation step and flush newly unblocked logs."""

        self._wandb_step_barriers.discard(int(step))
        self._flush_wandb_deferred_logs()

    def _should_defer_wandb_log(self, step: int) -> bool:
        return bool(self._wandb_step_barriers) and int(step) > min(self._wandb_step_barriers)

    def _flush_wandb_deferred_logs(self, *, force: bool = False) -> None:
        wandb_logger = self.logger.get("wandb")
        if wandb_logger is None or not self._wandb_deferred_logs:
            return

        if force or not self._wandb_step_barriers:
            max_unblocked_step = None
        else:
            max_unblocked_step = min(self._wandb_step_barriers)

        ready = []
        remaining = []
        for record in self._wandb_deferred_logs:
            if max_unblocked_step is None or record[0] <= max_unblocked_step:
                ready.append(record)
            else:
                remaining.append(record)
        self._wandb_deferred_logs = remaining

        # A later validation may finish before an earlier one. Sorting preserves
        # W&B's monotonic-step contract while retaining call order within a step.
        # W&B drops a second call to an already committed explicit step, so only
        # the final record for each step may commit the merged partial history.
        ordered = sorted(ready, key=lambda record: (record[0], record[1]))
        index = 0
        while index < len(ordered):
            group_end = index + 1
            log_step = ordered[index][0]
            while group_end < len(ordered) and ordered[group_end][0] == log_step:
                group_end += 1

            group_requests_commit = any(record[3] for record in ordered[index:group_end])
            step_is_still_blocked = log_step in self._wandb_step_barriers
            for record_index in range(index, group_end):
                _, _sequence, log_data, _requested_commit = ordered[record_index]
                is_last_in_group = record_index == group_end - 1
                commit = is_last_in_group and group_requests_commit and not step_is_still_blocked
                wandb_logger.log(
                    data=log_data,
                    step=int(log_step) + self._wandb_step_offset,
                    commit=commit,
                )
            index = group_end

    def define_metrics(self, metrics: str | list[str], step_metric: str | None = None, backend=None):
        if isinstance(metrics, str):
            metrics = [metrics]

        for default_backend, logger_instance in self.logger.items():
            if backend is not None and default_backend not in backend:
                continue
            if not hasattr(logger_instance, "define_metric"):
                continue

            for metric_name in metrics:
                effective_step_metric = step_metric
                if (
                    default_backend == "wandb"
                    and self._wandb_global_step_backfill
                    and step_metric is not None
                ):
                    effective_step_metric = "training/global_step"
                    step_cache_key = (default_backend, effective_step_metric, None)
                    if step_cache_key not in self._defined_metrics:
                        logger_instance.define_metric(effective_step_metric)
                        self._defined_metrics.add(step_cache_key)
                    if metric_name == effective_step_metric:
                        continue

                cache_key = (default_backend, metric_name, effective_step_metric)
                if cache_key in self._defined_metrics:
                    continue

                kwargs = {}
                if effective_step_metric is not None:
                    kwargs["step_metric"] = effective_step_metric
                logger_instance.define_metric(metric_name, **kwargs)
                self._defined_metrics.add(cache_key)

    def finish(self, exit_code: int = 0):
        if self._finished:
            return
        self._finished = True
        # Normal trainer shutdown releases every validation barrier first. This
        # fallback prevents unrelated training metrics from being lost if a run
        # exits early while an async validation result is still unavailable.
        try:
            self._wandb_step_barriers.clear()
            self._flush_wandb_deferred_logs(force=True)
        except BaseException:
            exit_code = exit_code or 1
            raise
        finally:
            if "wandb" in self.logger:
                self.logger["wandb"].finish(exit_code=exit_code)
            if "swanlab" in self.logger:
                self.logger["swanlab"].finish()
            if "vemlp_wandb" in self.logger:
                self.logger["vemlp_wandb"].finish(exit_code=exit_code)
            if "tensorboard" in self.logger:
                self.logger["tensorboard"].finish()
            if "clearml" in self.logger:
                self.logger["clearml"].finish()
            if "trackio" in self.logger:
                self.logger["trackio"].finish()
            if "file" in self.logger:
                self.logger["file"].finish()

    def __del__(self):
        try:
            # Only an explicit finish() confirms successful completion.
            self.finish(exit_code=1)
        except Exception:
            # Best-effort cleanup during interpreter teardown.
            pass


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "wb", buffering=0)

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get(
            "TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}"
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.
        self._consecutive_slashes_pattern = re.compile(r"/+")

    def log(self, data, step):
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")
            # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
            sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)
            if sanitized != key:
                self.logger.warning(
                    "[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized
                )
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


_VALIDATION_SAMPLE_COLUMNS = [
    "step",
    "sample_index",
    "uid",
    "data_source",
    "score",
    "reward",
    "correct",
    "acc",
    "all_passed",
    "passed_tests",
    "total_tests",
    "passed_tests_ratio",
    "incorrect_format",
    "format_reward",
    "timed_out",
    "error",
    "error_message",
    "pred",
    "ground_truth",
    "input",
    "output",
    "feedback",
    "extra_info",
]


def _coerce_python_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def _serialize_validation_sample_value(value: Any) -> Any:
    value = _coerce_python_scalar(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return repr(value)
    return str(value)


def _normalize_validation_sample(
    sample: Any, *, step: int | None = None, sample_index: int | None = None
) -> dict[str, Any]:
    if isinstance(sample, dict):
        raw_sample = dict(sample)
    else:
        raw_sample = {}
        if len(sample) > 0:
            raw_sample["input"] = sample[0]
        if len(sample) > 1:
            raw_sample["output"] = sample[1]
        if len(sample) > 2:
            raw_sample["score"] = sample[2]

    if step is not None:
        raw_sample.setdefault("step", step)
    if sample_index is not None:
        raw_sample.setdefault("sample_index", sample_index)

    if "ground_truth" not in raw_sample:
        if "gt" in raw_sample:
            raw_sample["ground_truth"] = raw_sample["gt"]
        elif "gts" in raw_sample:
            raw_sample["ground_truth"] = raw_sample["gts"]

    normalized = {column: None for column in _VALIDATION_SAMPLE_COLUMNS}
    for column in _VALIDATION_SAMPLE_COLUMNS:
        if column == "extra_info":
            continue
        if column in raw_sample:
            normalized[column] = _serialize_validation_sample_value(raw_sample[column])

    recognized_keys = set(_VALIDATION_SAMPLE_COLUMNS) | {"gt", "gts"}
    extras = {}
    for key in sorted(raw_sample):
        if key in recognized_keys:
            continue
        extras[key] = _serialize_validation_sample_value(raw_sample[key])
    normalized["extra_info"] = json.dumps(extras, ensure_ascii=False, sort_keys=True) if extras else ""
    return normalized


def normalize_validation_samples(samples: list[Any], *, step: int | None = None) -> list[dict[str, Any]]:
    return [
        _normalize_validation_sample(sample, step=step, sample_index=index)
        for index, sample in enumerate(samples)
    ]


def _validation_group_key(sample: dict[str, Any], *, fallback_index: int) -> tuple[str, str]:
    uid = sample.get("uid")
    if uid not in (None, ""):
        return ("uid", str(uid))

    input_text = sample.get("input")
    if input_text not in (None, ""):
        return ("input", str(input_text))

    return ("sample_index", str(sample.get("sample_index", fallback_index)))


def _validation_score_for_sorting(sample: dict[str, Any]) -> float:
    score = _coerce_python_scalar(sample.get("score"))
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(numeric_score):
        return float("-inf")
    return numeric_score


def select_diverse_validation_samples(
    samples: list[dict[str, Any]], limit: int, *, seed: int = 42
) -> list[dict[str, Any]]:
    if limit <= 0 or not samples:
        return []
    if len(samples) <= limit:
        return list(samples)

    grouped_samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, sample in enumerate(samples):
        grouped_samples.setdefault(_validation_group_key(sample, fallback_index=index), []).append(sample)

    group_keys = sorted(grouped_samples, key=lambda key: f"{key[0]}::{key[1]}")
    random.Random(seed).shuffle(group_keys)

    remaining_groups: list[list[dict[str, Any]]] = []
    for group_key in group_keys:
        members = list(grouped_samples[group_key])
        members.sort(
            key=lambda sample: (
                -_validation_score_for_sorting(sample),
                str(sample.get("output", "")),
                str(sample.get("input", "")),
            )
        )
        remaining_groups.append(members)

    selected: list[dict[str, Any]] = []
    while remaining_groups and len(selected) < limit:
        next_round_groups: list[list[dict[str, Any]]] = []
        for members in remaining_groups:
            if not members:
                continue
            selected.append(members.pop(0))
            if len(selected) >= limit:
                break
            if members:
                next_round_groups.append(members)
        remaining_groups = next_round_groups
    return selected


def select_random_uid_validation_samples(
    samples: list[dict[str, Any]], uid_limit: int, *, seed: int = 42
) -> list[dict[str, Any]]:
    if uid_limit <= 0 or not samples:
        return []

    grouped_samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, sample in enumerate(samples):
        grouped_samples.setdefault(_validation_group_key(sample, fallback_index=index), []).append(sample)

    group_keys = sorted(grouped_samples, key=lambda key: f"{key[0]}::{key[1]}")
    random.Random(seed).shuffle(group_keys)
    selected_keys = group_keys[: min(uid_limit, len(group_keys))]

    selected: list[dict[str, Any]] = []
    for group_key in selected_keys:
        selected.extend(grouped_samples[group_key])
    return selected


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if not samples:
            return

        normalized_samples = normalize_validation_samples(samples, step=step)

        if "wandb" in loggers:
            self.log_generations_to_wandb(normalized_samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(normalized_samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(normalized_samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(normalized_samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(normalized_samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(normalized_samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, _wandb_explicit_step(step), wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        columns = list(_VALIDATION_SAMPLE_COLUMNS)

        if not hasattr(self, "validation_table"):
            self.validation_table = wandb.Table(columns=columns)

        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        for sample in samples:
            row_data = [sample.get(column, None) for column in columns]
            new_table.add_data(*row_data)

        if wandb.run is not None:
            wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()
        headers = list(_VALIDATION_SAMPLE_COLUMNS)
        swanlab_rows = [[sample.get(column, None) for column in headers] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_rows)

        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                with open(validation_gen_step_file, "w") as file:
                    json.dump(samples, file, ensure_ascii=False)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(samples),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        text_content = f"**Generation Results - Step {step}**\n\n"

        for sample in samples:
            sample_index = sample.get("sample_index", 0)
            text_content += f"### Sample {sample_index}\n"
            text_content += f"**UID:** {sample.get('uid')}\n\n"
            text_content += f"**Data source:** {sample.get('data_source')}\n\n"
            text_content += f"**Score:** {sample.get('score')}\n\n"
            text_content += f"**Ground truth:** {sample.get('ground_truth')}\n\n"
            text_content += f"**Input:** {sample.get('input')}\n\n"
            text_content += f"**Output:** {sample.get('output')}\n\n"
            if sample.get("feedback"):
                text_content += f"**Feedback:** {sample.get('feedback')}\n\n"
            if sample.get("extra_info"):
                text_content += f"**Extra info:** {sample.get('extra_info')}\n\n"
            text_content += "---\n\n"

        self.writer.add_text("val/generations", text_content, step)
        self.writer.flush()
