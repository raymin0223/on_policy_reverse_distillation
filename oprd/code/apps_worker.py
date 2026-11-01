"""One APPS submission per process; official comparison semantics unchanged."""

import argparse
import importlib.util
import io
import json
import os
import resource
import signal
import sys
import time
import traceback

from oprd.code.common import OFFICIAL_APPS, sha256, text_hash
from oprd.code.isolation import restrict_apps_submission

OUTPUT_CAP = 16 * 1024**2


class OutputLimitExceeded(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    job = json.load(open(args.job))
    if sha256(OFFICIAL_APPS) != job["evaluator_sha256"]:
        raise RuntimeError("Official APPS testing_util.py changed")
    spec = importlib.util.spec_from_file_location("apps_official_testing", OFFICIAL_APPS)
    testing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(testing)
    assert testing.timeout == 4
    import numpy as np

    # The official stdin wrapper loads a module named tmp_sol, then invokes
    # code(). Complete scripts must see __name__ == '__main__' when invoked.
    # Keep submitted code byte-for-byte and leave all output comparisons intact.
    if job["input_output"].get("fn_name") is None:
        original_runtime_module = testing.RuntimeModule

        class ScriptRuntimeModule:
            @staticmethod
            def from_string(*args, **kwargs):
                module = original_runtime_module.from_string(*args, **kwargs)
                module.__dict__["__name__"] = "__main__"
                # pyext copies attributes into the module, but function globals
                # retain the original exec namespace.
                if hasattr(module, "code"):
                    module.code.__globals__["__name__"] = "__main__"
                return module

        testing.RuntimeModule = ScriptRuntimeModule

    consumed = 0
    timed_out = 0
    overflow = False
    output_tail = ""

    class CappedStringIO(io.StringIO):
        def fileno(self):
            # Official APPS enables faulthandler on stderr. Native diagnostic
            # writes go to the separately capped parent stderr pipe.
            return 2

        def write(self, text):
            nonlocal consumed, overflow, output_tail
            consumed += len(text.encode("utf-8", errors="replace"))
            if consumed > OUTPUT_CAP:
                overflow = True
                raise OutputLimitExceeded("stdout/stderr exceeded 16 MiB")
            output_tail = (output_tail + text[-4000:])[-4000:]
            return super().write(text)

        def writelines(self, lines):
            for line in lines:
                self.write(line)

        def seek(self, offset, whence=0):
            nonlocal overflow
            position = super().seek(offset, whence)
            if position > OUTPUT_CAP:
                overflow = True
                raise OutputLimitExceeded("stdout/stderr seek exceeded 16 MiB")
            return position

        def truncate(self, size=None):
            nonlocal overflow
            if size is not None and size > OUTPUT_CAP:
                overflow = True
                raise OutputLimitExceeded("stdout/stderr truncate exceeded 16 MiB")
            return super().truncate(size)

    def on_timeout(signum, frame):
        nonlocal timed_out
        timed_out += 1
        raise testing.TimeoutException()

    # Keep the official Capturing/custom_compare_/call-based equivalence code.
    # Only bound its in-memory capture and observe its existing timeout signal.
    testing.StringIO = CappedStringIO
    signal.signal(signal.SIGALRM, on_timeout)
    captured = CappedStringIO()
    fd = os.open(args.result, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    write = os.write
    start = time.monotonic()
    result = {
        "submitted_solution_sha256": text_hash(job["solution"]),
        "stdin_execution_name": "__main__" if job["input_output"].get("fn_name") is None else None,
        "adapter_version": 2,
    }
    try:
        restrict_apps_submission(args.cpu)
        result["observed_limits"] = {
            name: list(resource.getrlimit(getattr(resource, "RLIMIT_" + name)))
            for name in ("AS", "DATA", "STACK", "CPU", "FSIZE")
        }
        result["observed_cpu_affinity"] = sorted(os.sched_getaffinity(0))
        sys.stdout = captured
        sys.stderr = captured
        values = testing.run_test(problem={"input_output": job["input_output"]}, test=job["solution"])
        values = [bool(x) if isinstance(x, (bool, np.bool_)) else int(x) for x in values]
        total = len(job["input_output"]["inputs"])
        passed = len(values) == total and total > 0 and all(type(x) is bool and x for x in values)
        if passed:
            status = "pass"
        elif timed_out:
            status = "timeout"
        elif -2 in [x for x in values if type(x) is int]:
            status = "compile_error"
        elif -1 in [x for x in values if type(x) is int]:
            status = "runtime_error"
        else:
            status = "wrong_answer"
        if overflow:
            status = "runtime_error"
            passed = False
        result.update(
            status=status,
            reward=int(passed),
            test_results=values,
            passed_tests=sum(type(x) is bool and x for x in values),
            total_tests=total,
        )
    except OutputLimitExceeded:
        result.update(status="runtime_error", reward=0, error="output_limit")
    except BaseException as exc:
        result.update(
            status="runtime_error",
            reward=0,
            error=f"{type(exc).__name__}: {str(exc)[:1000]}",
            traceback=traceback.format_exc()[-4000:],
        )
    finally:
        signal.alarm(0)
        result.update(
            elapsed_seconds=time.monotonic() - start,
            per_test_timeouts=timed_out,
            output_limit_exceeded=overflow,
            captured_output_bytes=consumed,
            diagnostics=output_tail,
        )
        write(fd, (json.dumps(result) + "\n").encode())


if __name__ == "__main__":
    main()
