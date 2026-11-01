"""External wall deadline and pipe caps around each isolated APPS submission."""

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from oprd.code import common
from oprd.code.common import OFFICIAL_APPS, cpu_env, sha256

OUTPUT_CAP = 16 * 1024**2


def evaluate_submission(problem, solution, cpu=None):
    cpu = min(os.sched_getaffinity(0)) if cpu is None else cpu
    with tempfile.TemporaryDirectory(prefix="apps_submission_") as temporary:
        root = Path(temporary)
        job = root / "job.json"
        result_path = root / "result.json"
        job.write_text(
            json.dumps(
                dict(
                    input_output=problem["input_output"],
                    solution=solution,
                    evaluator_sha256=sha256(OFFICIAL_APPS),
                )
            )
        )
        start = time.monotonic()
        process = subprocess.Popen(
            [
                str(common.CPU_PYTHON),
                "-m",
                "oprd.code.apps_worker",
                "--job",
                str(job),
                "--result",
                str(result_path),
                "--cpu",
                str(cpu),
            ],
            cwd=root,
            env=cpu_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
        reason = None
        output_bytes = 0
        diagnostic = bytearray()
        with selectors.DefaultSelector() as selector:
            for pipe in (process.stdout, process.stderr):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() - start >= 10:
                    reason = "submission_wall_timeout"
                    break
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output_bytes += len(chunk)
                    diagnostic.extend(chunk[: max(0, 4000 - len(diagnostic))])
                    if output_bytes > OUTPUT_CAP:
                        reason = "output_limit"
                        break
                if reason:
                    break
        if reason:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=max(0.01, 10 - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reason = "submission_wall_timeout"
        finally:
            process.stdout.close()
            process.stderr.close()
        if reason:
            result = dict(
                status="timeout" if reason == "submission_wall_timeout" else "runtime_error",
                reward=0,
                error=reason,
            )
        elif result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                diagnostics = diagnostic.decode(errors="replace")
                if process.returncode == 0:
                    raise RuntimeError(
                        f"Invalid APPS worker result despite clean exit: {diagnostics!r}"
                    ) from exc
                # A submission may exhaust its memory before the worker can
                # serialize a result. This is one failed submission, not a
                # reason to abort every other task in the checkpoint.
                result = dict(
                    status="runtime_error",
                    reward=0,
                    error="memory_limit" if "MemoryError" in diagnostics else "worker_exit_invalid_result",
                    diagnostics=diagnostics,
                )
        else:
            result = dict(
                status="runtime_error",
                reward=0,
                error="worker_exit",
                diagnostics=diagnostic.decode(errors="replace"),
            )
        combined_output = output_bytes + result.get("captured_output_bytes", 0)
        if combined_output > OUTPUT_CAP:
            result.update(status="runtime_error", reward=0, error="output_limit", output_limit_exceeded=True)
        result.update(
            worker_returncode=process.returncode,
            wall_seconds=time.monotonic() - start,
            fd_output_bytes=output_bytes,
            combined_output_bytes=combined_output,
            cpu=cpu,
        )
        return result
