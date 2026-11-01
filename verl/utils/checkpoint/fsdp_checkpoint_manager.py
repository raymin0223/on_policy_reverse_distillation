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
import json
import logging
import os
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Optional
import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank
from .checkpoint_manager import BaseCheckpointManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
_ASYNC_MARKER_VERSION = 1
_DEFAULT_ASYNC_SAVE_TIMEOUT_SECONDS = 1800.0


def clone_checkpoint_payload_to_cpu(value: Any) -> Any:
    """Return an ownership-independent CPU snapshot of a tensor tree.

    Async writers must not retain aliases to live optimizer or auxiliary state:
    the next training step is allowed to mutate those objects as soon as the
    save RPC returns.  FSDP state extraction with ``offload_to_cpu=True`` moves
    shards to the host, and this additional clone gives the writer exclusive
    ownership.  The same helper handles optimizer and lag-state containers.
    """
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, dict):
        cloned = copy(value)
        cloned.clear()
        cloned.update(((key, clone_checkpoint_payload_to_cpu(item)) for key, item in value.items()))
        if hasattr(value, "_metadata"):
            cloned._metadata = clone_checkpoint_payload_to_cpu(value._metadata)
        return cloned
    if isinstance(value, list):
        return [clone_checkpoint_payload_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        items = [clone_checkpoint_payload_to_cpu(item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return type(value)(items)
    return deepcopy(value)


def fsync_parent_directory(path: str) -> None:
    """Make a completed atomic rename durable across a node/power failure."""
    directory = os.path.dirname(os.path.abspath(path))
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory_tree(root: str) -> None:
    """Durably flush a metadata tree before publishing its transaction."""
    for directory, _subdirectories, filenames in os.walk(root, topdown=False):
        for filename in filenames:
            with open(os.path.join(directory, filename), "rb") as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    fsync_parent_directory(root)


def atomic_torch_save(payload: Any, path: str) -> None:
    """Write one torch payload without exposing a partial final filename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        torch.save(payload, temporary_path)
        with open(temporary_path, "rb") as payload_file:
            os.fsync(payload_file.fileno())
        os.replace(temporary_path, path)
        fsync_parent_directory(path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")
        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self._async_save_executor: ThreadPoolExecutor | None = None
        self._async_save_future: Future | None = None
        self._async_save_abort_event: threading.Event | None = None

    def _async_save_enabled(self) -> bool:
        return bool(self.checkpoint_config and self.checkpoint_config.get("async_save", False))

    def _wait_async_save(self) -> None:
        if self._async_save_future is None:
            return
        future = self._async_save_future
        try:
            future.result()
        finally:
            if future.done() and self._async_save_future is future:
                self._async_save_future = None
                self._async_save_abort_event = None

    def _request_async_save_abort(self) -> None:
        """Wake this rank's background marker waits after a peer writer fails."""
        abort_event = getattr(self, "_async_save_abort_event", None)
        if abort_event is not None:
            abort_event.set()

    def _raise_if_async_save_abort_requested(self) -> None:
        abort_event = getattr(self, "_async_save_abort_event", None)
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError(
                "Asynchronous FSDP checkpoint save aborted after a writer failed on another rank."
            )

    def raise_if_async_save_failed_distributed(self) -> bool:
        """Nonblocking writer-health check called at a rollout boundary.

        A save can take many rollout steps to serialize.  Waiting until the
        next checkpoint would waste those rollouts after an early ENOSPC or
        writer exception.  Every actor rank calls this method together; the
        background threads themselves still perform no collectives.
        """
        local_error: BaseException | None = None
        future = self._async_save_future
        if future is not None and future.done():
            try:
                local_error = future.exception(timeout=0)
            except BaseException as exc:
                local_error = exc
        any_failed = local_error is not None
        all_done = future is None or future.done()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                backend = str(torch.distributed.get_backend()).lower()
                reduction_device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if "nccl" in backend
                    else torch.device("cpu")
                )
                status = torch.tensor(
                    [0 if any_failed else 1, 1 if all_done else 0], dtype=torch.int32, device=reduction_device
                )
                torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MIN)
                any_failed = not bool(status[0].item())
                all_done = bool(status[1].item())
            except BaseException:
                self._request_async_save_abort()
                raise
        if any_failed:
            self._request_async_save_abort()
            if future is not None and future.done():
                try:
                    self._wait_async_save()
                except BaseException as observed_error:
                    if local_error is None:
                        local_error = observed_error
            if local_error is not None:
                raise RuntimeError(
                    f"Asynchronous FSDP checkpoint writer failed on rank {self.rank}: {local_error}"
                ) from local_error
            raise RuntimeError("An asynchronous FSDP checkpoint writer failed on another rank.")
        if all_done:
            self._wait_async_save()
            return True
        return False

    def wait_async_save_distributed(self) -> None:
        """Wait in the caller thread and propagate a writer failure to every rank.

        Background checkpoint threads deliberately perform no collectives.  All
        worker caller threads poll their local futures and reduce the status
        together.  This is intentionally different from blocking on each local
        ``Future`` first: a failed rank might be unable to write its filesystem
        failure marker, in which case a peer rank-0 writer would otherwise wait
        until the full transaction timeout before reaching the collective.
        """
        while not self.raise_if_async_save_failed_distributed():
            time.sleep(0.1)

    def _raise_if_distributed_staging_failed(
        self, local_error: BaseException | None, *, context: str
    ) -> None:
        """Synchronize caller-thread staging failures before any rank advances."""
        any_failed = local_error is not None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = str(torch.distributed.get_backend()).lower()
            reduction_device = (
                torch.device("cuda", torch.cuda.current_device())
                if "nccl" in backend
                else torch.device("cpu")
            )
            staging_ok = torch.tensor(0 if any_failed else 1, dtype=torch.int32, device=reduction_device)
            torch.distributed.all_reduce(staging_ok, op=torch.distributed.ReduceOp.MIN)
            any_failed = not bool(staging_ok.item())
        if not any_failed:
            return
        if local_error is not None:
            raise RuntimeError(f"{context} failed on rank {self.rank}: {local_error}") from local_error
        raise RuntimeError(f"{context} failed on another rank.")

    def _submit_async_save(self, fn, *, global_step: int) -> None:
        if self._async_save_future is not None:
            raise RuntimeError(
                "Cannot submit an FSDP checkpoint while a previous asynchronous save is pending."
            )
        if self._async_save_executor is None:
            self._async_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fsdp-ckpt-save")
        self._async_save_abort_event = threading.Event()
        try:
            self._async_save_future = self._async_save_executor.submit(fn)
        except BaseException:
            self._async_save_abort_event.set()
            self._async_save_abort_event = None
            raise

        def _log_async_save_done(future: Future) -> None:
            exc = future.exception()
            if exc is None:
                log_with_rank(
                    f"Async FSDP checkpoint save finished for step {global_step}",
                    rank=self.rank,
                    logger=logger,
                )
            else:
                log_with_rank(
                    f"Async FSDP checkpoint save failed for step {global_step}: {exc}",
                    rank=self.rank,
                    logger=logger,
                )

        self._async_save_future.add_done_callback(_log_async_save_done)

    @staticmethod
    def _atomic_write_text(path: str, value: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            fsync_parent_directory(path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @classmethod
    def _atomic_write_json(cls, path: str, payload: dict[str, Any]) -> None:
        cls._atomic_write_text(path, json.dumps(payload, sort_keys=True))

    @staticmethod
    def _checkpoint_step_and_root(local_path: str, global_step: int) -> tuple[str, str]:
        """Resolve ``global_step_N`` and its parent for actor or direct paths."""
        cursor = os.path.abspath(local_path.rstrip(os.sep))
        expected_name = f"global_step_{int(global_step)}"
        while os.path.basename(cursor) != expected_name:
            parent = os.path.dirname(cursor)
            if parent == cursor:
                raise ValueError(
                    f"Checkpoint path {local_path!r} has no {expected_name!r} ancestor; refusing to publish an ambiguous latest tracker."
                )
            cursor = parent
        return (cursor, os.path.dirname(cursor))

    def _marker_path(self, local_path: str, kind: str, rank: int | None = None) -> str:
        suffix = "" if rank is None else f"_world_size_{self.world_size}_rank_{rank}"
        return os.path.join(local_path, f".{kind}{suffix}.json")

    @staticmethod
    def _validate_complete_file(path: str, *, label: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {label}: {path}")
        if os.path.getsize(path) <= 0:
            raise RuntimeError(f"Empty {label}: {path}")

    def _async_timeout_seconds(self) -> float:
        if self.checkpoint_config is None:
            return _DEFAULT_ASYNC_SAVE_TIMEOUT_SECONDS
        return float(
            self.checkpoint_config.get("async_save_timeout_seconds", _DEFAULT_ASYNC_SAVE_TIMEOUT_SECONDS)
        )

    def _read_failure_marker(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
            return str(payload.get("error", "unknown asynchronous checkpoint failure"))
        except Exception:
            return "unreadable asynchronous checkpoint failure marker"

    def _wait_for_marker(
        self, path: str, *, label: str, failure_paths: list[str], timeout_seconds: float
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            for failure_path in failure_paths:
                if os.path.isfile(failure_path):
                    raise RuntimeError(f"{label} failed: {self._read_failure_marker(failure_path)}")
            self._raise_if_async_save_abort_requested()
            if os.path.isfile(path):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out after {timeout_seconds:.1f}s waiting for {label}: {path}")
            time.sleep(0.1)

    def _write_failure_marker(self, local_path: str, exc: BaseException, *, worker: bool = False) -> None:
        kind = "worker_sidecars_failed" if worker else "async_checkpoint_failed"
        marker = self._marker_path(local_path, kind, self.rank)
        self._atomic_write_json(
            marker,
            {
                "version": _ASYNC_MARKER_VERSION,
                "world_size": int(self.world_size),
                "rank": int(self.rank),
                "error": f"{type(exc).__name__}: {exc}"[:4096],
            },
        )

    def prepare_async_transaction(self, local_path: str) -> None:
        """Clear this rank's stale private markers before state snapshotting."""
        marker_kinds = (
            "async_checkpoint_started",
            "worker_sidecars_ready",
            "worker_sidecars_failed",
            "async_checkpoint_ready",
            "async_checkpoint_failed",
        )
        for kind in marker_kinds:
            marker = self._marker_path(local_path, kind, self.rank)
            if os.path.exists(marker):
                os.remove(marker)
        if self.rank == 0:
            trainer_failure_marker = os.path.join(local_path, ".trainer_sidecar_failed.json")
            if os.path.exists(trainer_failure_marker):
                os.remove(trainer_failure_marker)
            complete_marker = self._marker_path(local_path, "checkpoint_complete")
            if os.path.exists(complete_marker):
                os.remove(complete_marker)
        fsync_parent_directory(os.path.join(local_path, ".transaction_cleanup"))

    def mark_async_transaction_started(self, local_path: str, *, global_step: int) -> None:
        """Durably publish transaction intent before any large payload is staged.

        Without this caller-thread marker, a crash after creating the actor
        directory but before the background writer's first ready/failure marker
        leaves a markerless partial checkpoint.  Explicit resume could then
        misclassify it as a legacy synchronous checkpoint.
        """
        self._atomic_write_json(
            self._marker_path(local_path, "async_checkpoint_started", self.rank),
            {
                "version": _ASYNC_MARKER_VERSION,
                "world_size": int(self.world_size),
                "rank": int(self.rank),
                "global_step": int(global_step),
            },
        )

    def _reject_committed_path_overwrite(self, local_path: str, *, global_step: int) -> None:
        """Never overwrite a path already published as a complete transaction."""
        complete_marker = self._marker_path(local_path, "checkpoint_complete")
        step_dir, checkpoint_root = self._checkpoint_step_and_root(local_path, global_step)
        tracker_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
        tracker_step: int | None = None
        if os.path.isfile(tracker_path):
            try:
                with open(tracker_path, encoding="utf-8") as tracker_file:
                    tracker_step = int(tracker_file.read().strip())
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot safely classify checkpoint path because its root tracker is unreadable: {tracker_path}"
                ) from exc
        has_complete_marker = os.path.isfile(complete_marker)
        if tracker_step is None or tracker_step < int(global_step):
            return
        raise FileExistsError(
            f"Refusing to overwrite an already committed checkpoint path in place: path={step_dir}, checkpoint_step={global_step}, complete_marker={has_complete_marker}, root_tracker_step={tracker_step}."
        )

    def _is_same_step_already_committed(self, local_path: str, *, global_step: int) -> bool:
        """True when this exact step is already committed (an idempotent re-save).

        A save request for a step whose checkpoint is already the committed latest
        (root tracker == global_step) is a benign no-op: the payload is fully on
        disk (the tracker is written last, so tracker == step ⇒ committed).  This
        happens e.g. when a best-validation candidate checkpoint is scheduled at the
        exact step we just resumed from, or when the same validation step re-triggers
        a save.  Such a save should be skipped, not fatally rejected — which is the
        "same-step retry is safe" case the anti-overwrite guard's docstring intends.
        A strictly-older overwrite (tracker > step) is NOT this case and still routes
        to _reject_committed_path_overwrite.
        """
        _, checkpoint_root = self._checkpoint_step_and_root(local_path, global_step)
        tracker_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(tracker_path):
            return False
        try:
            with open(tracker_path, encoding="utf-8") as tracker_file:
                tracker_step = int(tracker_file.read().strip())
        except (OSError, ValueError):
            return False
        return tracker_step == int(global_step)

    def _all_ranks_skip_same_step(self, local_skip: bool) -> bool:
        """Collectively agree on an idempotent same-step skip.

        Every rank shares the same root tracker file, so they reach the same
        decision; the all-reduce (MIN) makes the skip unanimous regardless of any
        filesystem-visibility skew, so either every rank returns early together or
        none does — never a mix that would deadlock the post-setup barrier.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local_skip
        backend = str(torch.distributed.get_backend()).lower()
        reduction_device = (
            torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else torch.device("cpu")
        )
        skip_all = torch.tensor(1 if local_skip else 0, dtype=torch.int32, device=reduction_device)
        torch.distributed.all_reduce(skip_all, op=torch.distributed.ReduceOp.MIN)
        return bool(skip_all.item())

    def mark_worker_sidecars_complete(
        self, local_path: str, *, global_step: int, required_paths: list[str] | None = None
    ) -> None:
        """Publish the caller-thread sidecar manifest after synchronous extras land."""
        required_paths = list(required_paths or [])
        local_root = os.path.abspath(local_path)
        relative_paths: list[str] = []
        for path in required_paths:
            absolute_path = os.path.abspath(path)
            if os.path.commonpath((local_root, absolute_path)) != local_root:
                raise ValueError(f"Worker checkpoint sidecar escapes actor directory: {path}")
            self._validate_complete_file(absolute_path, label="worker checkpoint sidecar")
            relative_paths.append(os.path.relpath(absolute_path, local_root))
        self._atomic_write_json(
            self._marker_path(local_path, "worker_sidecars_ready", self.rank),
            {
                "version": _ASYNC_MARKER_VERSION,
                "world_size": int(self.world_size),
                "rank": int(self.rank),
                "global_step": int(global_step),
                "required_paths": relative_paths,
            },
        )

    def mark_worker_sidecars_failed(self, local_path: str, exc: BaseException) -> None:
        self._write_failure_marker(local_path, exc, worker=True)

    def _validate_worker_sidecar_marker(self, local_path: str, *, global_step: int) -> None:
        marker = self._marker_path(local_path, "worker_sidecars_ready", self.rank)
        with open(marker, encoding="utf-8") as stream:
            payload = json.load(stream)
        expected = (_ASYNC_MARKER_VERSION, int(self.world_size), int(self.rank), int(global_step))
        actual = (
            payload.get("version"),
            int(payload.get("world_size", -1)),
            int(payload.get("rank", -1)),
            int(payload.get("global_step", -1)),
        )
        if actual != expected:
            raise RuntimeError(
                f"Invalid worker-sidecar marker {marker}: expected={expected}, actual={actual}"
            )
        for relative_path in payload.get("required_paths", []):
            absolute_path = os.path.abspath(os.path.join(local_path, relative_path))
            if os.path.commonpath((os.path.abspath(local_path), absolute_path)) != os.path.abspath(
                local_path
            ):
                raise RuntimeError(f"Worker-sidecar marker contains an escaping path: {relative_path!r}")
            self._validate_complete_file(absolute_path, label="worker checkpoint sidecar")

    def _save_rank0_hf_metadata(self, local_path: str):
        """Synchronously save small Hugging Face metadata and return its configs."""
        if fsdp_version(self.model) == 1:
            unwrap_model = self.model._fsdp_wrapped_module
        else:
            unwrap_model = self.model
        hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
        local_mkdir_safe(hf_config_tokenizer_path)
        model_config = unwrap_model.config
        generation_config = None
        if (
            unwrap_model.can_generate()
            and hasattr(model_config, "name_or_path")
            and model_config.name_or_path
        ):
            try:
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(hf_config_tokenizer_path)
            except Exception:
                pass
        if hasattr(model_config, "auto_map") and None in model_config.auto_map:
            model_config.auto_map = {
                key: value for key, value in model_config.auto_map.items() if key is not None
            }
        model_config.save_pretrained(hf_config_tokenizer_path)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(hf_config_tokenizer_path)
        log_with_rank(
            f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
        if hasattr(model_config, "auto_map"):
            custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)
        fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
        fsdp_config = FSDPConfig(FSDP_version=fsdp_version(self.model), world_size=self.world_size)
        with open(fsdp_config_path, "w", encoding="utf-8") as config_file:
            json.dump(asdict(fsdp_config), config_file, indent=4)
        fsync_directory_tree(hf_config_tokenizer_path)
        with open(fsdp_config_path, "rb") as config_file:
            os.fsync(config_file.fileno())
        fsync_parent_directory(fsdp_config_path)
        return (model_config, generation_config)

    @staticmethod
    def _global_step_from_checkpoint_path(local_path: str) -> int:
        cursor = os.path.abspath(local_path.rstrip(os.sep))
        while True:
            name = os.path.basename(cursor)
            if name.startswith("global_step_"):
                try:
                    return int(name.removeprefix("global_step_"))
                except ValueError as exc:
                    raise ValueError(f"Malformed checkpoint step directory: {cursor}") from exc
            parent = os.path.dirname(cursor)
            if parent == cursor:
                raise ValueError(f"Checkpoint path has no global_step_N ancestor: {local_path}")
            cursor = parent

    def _validate_async_transaction_for_load(self, local_path: str) -> bool:
        """Validate a transaction-marked checkpoint before loading large shards.

        Markerless checkpoints are retained for backward compatibility.  Once
        any async-transaction marker is present, however, the complete marker,
        root tracker, rank manifest, and every manifest payload are mandatory.
        A committed checkpoint with a pruned optimizer is not resumable.
        """
        if not os.path.isdir(local_path):
            raise FileNotFoundError(f"Checkpoint actor directory does not exist: {local_path}")
        complete_marker = self._marker_path(local_path, "checkpoint_complete")
        marker_names = [name for name in os.listdir(local_path) if name.startswith(".")]
        async_marker_present = any(
            (
                name.startswith((".async_checkpoint_", ".worker_sidecars_", ".trainer_sidecar_"))
                for name in marker_names
            )
        )
        if not os.path.isfile(complete_marker):
            if async_marker_present:
                raise RuntimeError(
                    f"Refusing to load an incomplete asynchronous checkpoint transaction: {local_path}"
                )
            return False
        global_step = self._global_step_from_checkpoint_path(local_path)
        with open(complete_marker, encoding="utf-8") as stream:
            complete_payload = json.load(stream)
        expected_complete = (_ASYNC_MARKER_VERSION, int(self.world_size), int(global_step))
        actual_complete = (
            complete_payload.get("version"),
            int(complete_payload.get("world_size", -1)),
            int(complete_payload.get("global_step", -1)),
        )
        if actual_complete != expected_complete:
            raise RuntimeError(
                f"Invalid asynchronous checkpoint complete marker {complete_marker}: expected={expected_complete}, actual={actual_complete}"
            )
        step_dir, checkpoint_root = self._checkpoint_step_and_root(local_path, global_step)
        tracker_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
        try:
            with open(tracker_path, encoding="utf-8") as stream:
                tracker_step = int(stream.read().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Committed async checkpoint has no valid root tracker: {tracker_path}"
            ) from exc
        if tracker_step < global_step:
            raise RuntimeError(
                f"Async checkpoint payloads are ready but the transaction was never committed: checkpoint_step={global_step}, root_tracker_step={tracker_step}, path={step_dir}"
            )
        ready_marker = self._marker_path(local_path, "async_checkpoint_ready", self.rank)
        with open(ready_marker, encoding="utf-8") as stream:
            rank_payload = json.load(stream)
        expected_rank = (_ASYNC_MARKER_VERSION, int(self.world_size), int(self.rank), int(global_step))
        actual_rank = (
            rank_payload.get("version"),
            int(rank_payload.get("world_size", -1)),
            int(rank_payload.get("rank", -1)),
            int(rank_payload.get("global_step", -1)),
        )
        if actual_rank != expected_rank:
            raise RuntimeError(
                f"Invalid rank checkpoint marker {ready_marker}: expected={expected_rank}, actual={actual_rank}"
            )
        for relative_path in rank_payload.get("written_paths", []):
            payload_path = os.path.abspath(os.path.join(local_path, relative_path))
            if os.path.commonpath((os.path.abspath(local_path), payload_path)) != os.path.abspath(local_path):
                raise RuntimeError(f"Rank checkpoint marker contains an escaping path: {relative_path!r}")
            self._validate_complete_file(payload_path, label=f"rank {self.rank} checkpoint payload")
        return True

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return
        self.wait_async_save_distributed()
        if self.should_load_model:
            assert self.model is not None, (
                "model must be provided when checkpoint_contents.load includes ['model']"
            )
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )
        transaction_error: BaseException | None = None
        try:
            self._validate_async_transaction_for_load(local_path)
        except BaseException as exc:
            transaction_error = exc
        self._raise_if_distributed_staging_failed(
            transaction_error, context="Checkpoint transaction preflight"
        )
        loaded_local_paths: list[str] = []
        model_state_dict = optimizer_state_dict = extra_state_dict = None
        preload_error: BaseException | None = None
        try:
            if self.should_load_model:
                remote_model_path = os.path.join(
                    local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                local_model_path = copy_to_local(remote_model_path)
                self._validate_complete_file(local_model_path, label="model checkpoint shard")
                model_state_dict = torch.load(local_model_path, map_location="cpu", weights_only=False)
                loaded_local_paths.append(local_model_path)
            if self.should_load_optimizer:
                remote_optim_path = os.path.join(
                    local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                local_optim_path = copy_to_local(remote_optim_path)
                self._validate_complete_file(local_optim_path, label="optimizer checkpoint shard")
                optimizer_state_dict = torch.load(local_optim_path, map_location="cpu", weights_only=False)
                loaded_local_paths.append(local_optim_path)
            if self.should_load_extra:
                remote_extra_state_path = os.path.join(
                    local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                local_extra_state_path = copy_to_local(remote_extra_state_path)
                self._validate_complete_file(local_extra_state_path, label="extra-state checkpoint shard")
                extra_state_dict = torch.load(local_extra_state_path, map_location="cpu", weights_only=False)
                loaded_local_paths.append(local_extra_state_path)
        except BaseException as exc:
            preload_error = exc
        self._raise_if_distributed_staging_failed(preload_error, context="Checkpoint shard preload")
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        apply_error: BaseException | None = None
        try:
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                if self.should_load_model:
                    assert model_state_dict is not None
                    self.model.load_state_dict(model_state_dict)
                    log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)
                if self.should_load_optimizer:
                    assert optimizer_state_dict is not None
                    self.optimizer.load_state_dict(optimizer_state_dict)
                    log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)
        except BaseException as exc:
            apply_error = exc
        self._raise_if_distributed_staging_failed(apply_error, context="Checkpoint model/optimizer restore")
        model_state_dict = None
        optimizer_state_dict = None
        extra_apply_error: BaseException | None = None
        try:
            if self.should_load_extra:
                assert extra_state_dict is not None
                if "rng" in extra_state_dict:
                    self.load_rng_state(extra_state_dict["rng"])
                    log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)
                lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
                if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                    log_with_rank(
                        f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger
                    )
        except BaseException as exc:
            extra_apply_error = exc
        self._raise_if_distributed_staging_failed(extra_apply_error, context="Checkpoint extra-state restore")
        extra_state_dict = None
        if del_local_after_load:
            for loaded_path in loaded_local_paths:
                try:
                    if is_non_local(loaded_path):
                        os.remove(loaded_path)
                except Exception as exc:
                    log_with_rank(
                        f"remove local resume ckpt file after loading failed, exception {exc} will be ignored",
                        rank=self.rank,
                        logger=logger,
                    )
        torch.distributed.barrier()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        exclude_from_max_ckpt_to_keep: bool = False,
        additional_payloads: list[tuple[Any, str, str]] | None = None,
        wait_for_worker_sidecars: bool = False,
        wait_for_trainer_sidecar: bool = False,
    ):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return
        self.wait_async_save_distributed()
        async_save = self._async_save_enabled()
        additional_payloads = list(additional_payloads or [])
        self.previous_global_step = global_step
        if async_save and self._all_ranks_skip_same_step(
            self._is_same_step_already_committed(local_path, global_step=global_step)
        ):
            log_with_rank(
                f"Checkpoint for step {global_step} is already committed; skipping redundant save.",
                rank=self.rank,
                logger=logger,
            )
            return
        setup_error: BaseException | None = None
        try:
            if async_save:
                self._reject_committed_path_overwrite(local_path, global_step=global_step)
            if self.rank == 0:
                self.ensure_checkpoint_capacity(max_ckpt_to_keep)
            local_path = local_mkdir_safe(local_path)
            if async_save:
                if self.rank == 0 and wait_for_trainer_sidecar:
                    step_dir, _ = self._checkpoint_step_and_root(local_path, global_step)
                    stale_data_path = os.path.join(step_dir, "data.pt")
                    if os.path.exists(stale_data_path):
                        os.remove(stale_data_path)
                        fsync_parent_directory(stale_data_path)
                self.prepare_async_transaction(local_path)
                self.mark_async_transaction_started(local_path, global_step=global_step)
        except BaseException as exc:
            setup_error = exc
        if async_save:
            self._raise_if_distributed_staging_failed(setup_error, context="Async checkpoint setup")
        elif setup_error is not None:
            raise setup_error
        torch.distributed.barrier()
        if self.should_save_model:
            assert self.model is not None, (
                "model must be provided when checkpoint_contents.save includes ['model']"
            )
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        async_save_payloads = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(
                    local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                optim_path = os.path.join(
                    local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                extra_path = os.path.join(
                    local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                if self.should_save_model:
                    if async_save:
                        model_staging_error: BaseException | None = None
                        try:
                            model_state_dict = self.model.state_dict()
                            model_state_dict = clone_checkpoint_payload_to_cpu(model_state_dict)
                        except BaseException as exc:
                            model_staging_error = exc
                        self._raise_if_distributed_staging_failed(
                            model_staging_error, context="Async model-state snapshot"
                        )
                        async_save_payloads.append((model_state_dict, model_path, "model"))
                    else:
                        model_state_dict = self.model.state_dict()
                        atomic_torch_save(model_state_dict, model_path)
                        log_with_rank(
                            f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger
                        )
                if self.should_save_optimizer:
                    if async_save:
                        optimizer_staging_error: BaseException | None = None
                        try:
                            optimizer_state_dict = self.optimizer.state_dict()
                            optimizer_state_dict = clone_checkpoint_payload_to_cpu(optimizer_state_dict)
                        except BaseException as exc:
                            optimizer_staging_error = exc
                        self._raise_if_distributed_staging_failed(
                            optimizer_staging_error, context="Async optimizer-state snapshot"
                        )
                        async_save_payloads.append((optimizer_state_dict, optim_path, "optim"))
                    else:
                        optimizer_state_dict = self.optimizer.state_dict()
                        atomic_torch_save(optimizer_state_dict, optim_path)
                        log_with_rank(
                            f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger
                        )
                if self.should_save_extra:
                    if async_save:
                        extra_staging_error: BaseException | None = None
                        try:
                            lr_scheduler_state_dict = (
                                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                            )
                            extra_state_dict = {
                                "lr_scheduler": lr_scheduler_state_dict,
                                "rng": self.get_rng_state(),
                            }
                            extra_state_dict = clone_checkpoint_payload_to_cpu(extra_state_dict)
                        except BaseException as exc:
                            extra_staging_error = exc
                        self._raise_if_distributed_staging_failed(
                            extra_staging_error, context="Async extra-state snapshot"
                        )
                        async_save_payloads.append((extra_state_dict, extra_path, "extra_state"))
                    else:
                        lr_scheduler_state_dict = (
                            self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                        )
                        extra_state_dict = {
                            "lr_scheduler": lr_scheduler_state_dict,
                            "rng": self.get_rng_state(),
                        }
                        atomic_torch_save(extra_state_dict, extra_path)
                        log_with_rank(
                            f"Saved extra_state to {os.path.abspath(extra_path)}",
                            rank=self.rank,
                            logger=logger,
                        )
        if async_save:
            additional_staging_error: BaseException | None = None
            staged_additional_payloads = []
            try:
                for payload, path, label in additional_payloads:
                    absolute_path = os.path.abspath(path)
                    if os.path.commonpath((os.path.abspath(local_path), absolute_path)) != os.path.abspath(
                        local_path
                    ):
                        raise ValueError(f"Additional checkpoint payload escapes actor directory: {path}")
                    staged_additional_payloads.append(
                        (clone_checkpoint_payload_to_cpu(payload), absolute_path, str(label))
                    )
            except BaseException as exc:
                additional_staging_error = exc
            self._raise_if_distributed_staging_failed(
                additional_staging_error, context="Async additional-payload snapshot"
            )
            async_save_payloads.extend(staged_additional_payloads)
        else:
            for payload, path, label in additional_payloads:
                absolute_path = os.path.abspath(path)
                if os.path.commonpath((os.path.abspath(local_path), absolute_path)) != os.path.abspath(
                    local_path
                ):
                    raise ValueError(f"Additional checkpoint payload escapes actor directory: {path}")
                atomic_torch_save(payload, absolute_path)
                log_with_rank(f"Saved {label} to {absolute_path}", rank=self.rank, logger=logger)
        hf_metadata_error: BaseException | None = None
        model_config = generation_config = None
        try:
            if self.rank == 0:
                model_config, generation_config = self._save_rank0_hf_metadata(local_path)
        except BaseException as exc:
            hf_metadata_error = exc
        if async_save:
            self._raise_if_distributed_staging_failed(
                hf_metadata_error, context="Async Hugging Face metadata snapshot"
            )
        elif hf_metadata_error is not None:
            raise hf_metadata_error
        if not async_save:
            torch.distributed.barrier()
        if self.should_save_hf_model:
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
            hf_model_error: BaseException | None = None
            if self.rank == 0:
                try:
                    hf_local_path = os.path.join(local_path, "huggingface")
                    os.makedirs(hf_local_path, exist_ok=True)
                    if "ForTokenClassification" in model_config.architectures[0]:
                        from transformers import AutoModelForTokenClassification

                        auto_model_cls = AutoModelForTokenClassification
                    elif "ForCausalLM" in model_config.architectures[0]:
                        from transformers import AutoModelForCausalLM

                        auto_model_cls = AutoModelForCausalLM
                    elif "ForConditionalGeneration" in model_config.architectures[0]:
                        import transformers
                        from packaging import version

                        if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                            from transformers import AutoModelForImageTextToText

                            auto_model_cls = AutoModelForImageTextToText
                        else:
                            from transformers import AutoModelForVision2Seq

                            auto_model_cls = AutoModelForVision2Seq
                    else:
                        raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")
                    with init_empty_weights():
                        save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                    save_model.to_empty(device="cpu")
                    if save_model.can_generate():
                        if generation_config is not None:
                            save_model.generation_config = generation_config
                        else:
                            print(
                                f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model."
                            )
                    save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                    fsync_directory_tree(hf_local_path)
                    log_with_rank(
                        f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )
                except BaseException as exc:
                    hf_model_error = exc
                finally:
                    state_dict = None
                    if "save_model" in locals():
                        del save_model
            self._raise_if_distributed_staging_failed(
                hf_model_error, context="Hugging Face full-model serialization"
            )
            state_dict = None

        def finalize_save() -> None:
            if not async_save:
                if self.rank == 0:
                    self.register_checkpoint(
                        local_path,
                        max_ckpt_to_keep,
                        exclude_from_max_ckpt_to_keep=exclude_from_max_ckpt_to_keep,
                    )
                return
            timeout_seconds = self._async_timeout_seconds()
            try:
                written_paths: list[str] = []
                while async_save_payloads:
                    self._raise_if_async_save_abort_requested()
                    state_dict, path, label = async_save_payloads.pop(0)
                    try:
                        atomic_torch_save(state_dict, path)
                    finally:
                        state_dict = None
                    written_paths.append(os.path.abspath(path))
                    log_with_rank(f"Saved {label} to {os.path.abspath(path)}", rank=self.rank, logger=logger)
                if self.rank == 0:
                    fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
                    self._validate_complete_file(fsdp_config_path, label="FSDP checkpoint metadata")
                    written_paths.append(os.path.abspath(fsdp_config_path))
                    hf_metadata_root = os.path.join(local_path, "huggingface")
                    for directory, _subdirectories, filenames in os.walk(hf_metadata_root):
                        for filename in filenames:
                            metadata_path = os.path.join(directory, filename)
                            self._validate_complete_file(
                                metadata_path, label="Hugging Face checkpoint metadata"
                            )
                            written_paths.append(os.path.abspath(metadata_path))
                if wait_for_worker_sidecars:
                    worker_ready = self._marker_path(local_path, "worker_sidecars_ready", self.rank)
                    self._wait_for_marker(
                        worker_ready,
                        label=f"rank {self.rank} worker checkpoint sidecars",
                        failure_paths=[self._marker_path(local_path, "worker_sidecars_failed", self.rank)],
                        timeout_seconds=timeout_seconds,
                    )
                    self._validate_worker_sidecar_marker(local_path, global_step=global_step)
                self._raise_if_async_save_abort_requested()
                ready_marker = self._marker_path(local_path, "async_checkpoint_ready", self.rank)
                self._atomic_write_json(
                    ready_marker,
                    {
                        "version": _ASYNC_MARKER_VERSION,
                        "world_size": int(self.world_size),
                        "rank": int(self.rank),
                        "global_step": int(global_step),
                        "written_paths": [os.path.relpath(path, local_path) for path in written_paths],
                    },
                )
                if self.rank != 0:
                    return
                all_failure_markers: list[str] = []
                for rank in range(self.world_size):
                    all_failure_markers.extend(
                        (
                            self._marker_path(local_path, "async_checkpoint_failed", rank),
                            self._marker_path(local_path, "worker_sidecars_failed", rank),
                        )
                    )
                all_failure_markers.append(os.path.join(local_path, ".trainer_sidecar_failed.json"))
                for rank in range(self.world_size):
                    rank_ready = self._marker_path(local_path, "async_checkpoint_ready", rank)
                    self._wait_for_marker(
                        rank_ready,
                        label=f"rank {rank} asynchronous checkpoint",
                        failure_paths=all_failure_markers,
                        timeout_seconds=timeout_seconds,
                    )
                    with open(rank_ready, encoding="utf-8") as stream:
                        rank_payload = json.load(stream)
                    expected_rank_header = (
                        _ASYNC_MARKER_VERSION,
                        int(self.world_size),
                        int(rank),
                        int(global_step),
                    )
                    actual_rank_header = (
                        rank_payload.get("version"),
                        int(rank_payload.get("world_size", -1)),
                        int(rank_payload.get("rank", -1)),
                        int(rank_payload.get("global_step", -1)),
                    )
                    if actual_rank_header != expected_rank_header:
                        raise RuntimeError(
                            f"Invalid rank checkpoint marker {rank_ready}: expected={expected_rank_header}, actual={actual_rank_header}"
                        )
                    for relative_path in rank_payload.get("written_paths", []):
                        payload_path = os.path.abspath(os.path.join(local_path, relative_path))
                        if os.path.commonpath((os.path.abspath(local_path), payload_path)) != os.path.abspath(
                            local_path
                        ):
                            raise RuntimeError(
                                f"Rank checkpoint marker contains an escaping path: {relative_path!r}"
                            )
                        self._validate_complete_file(payload_path, label=f"rank {rank} checkpoint payload")
                step_dir, checkpoint_root = self._checkpoint_step_and_root(local_path, global_step)
                if wait_for_trainer_sidecar:
                    data_path = os.path.join(step_dir, "data.pt")
                    self._wait_for_marker(
                        data_path,
                        label="trainer dataloader sidecar",
                        failure_paths=all_failure_markers,
                        timeout_seconds=timeout_seconds,
                    )
                    self._validate_complete_file(data_path, label="trainer dataloader sidecar")
                self._raise_if_async_save_abort_requested()
                self._atomic_write_json(
                    self._marker_path(local_path, "checkpoint_complete"),
                    {
                        "version": _ASYNC_MARKER_VERSION,
                        "world_size": int(self.world_size),
                        "global_step": int(global_step),
                    },
                )
                latest_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
                self._atomic_write_text(latest_path, str(global_step))
            except BaseException as exc:
                async_save_payloads.clear()
                try:
                    self._write_failure_marker(local_path, exc)
                except Exception:
                    logger.exception("Failed to publish asynchronous FSDP checkpoint failure marker")
                raise
            try:
                self.register_checkpoint(
                    local_path, max_ckpt_to_keep, exclude_from_max_ckpt_to_keep=exclude_from_max_ckpt_to_keep
                )
            except Exception:
                logger.exception(
                    "Checkpoint step %s committed successfully, but post-commit retention failed", global_step
                )

        if async_save:
            self._submit_async_save(finalize_save, global_step=global_step)
        else:
            finalize_save()
