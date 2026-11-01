"""Clean subprocess entry point for untrusted DeepCoder test execution."""

from __future__ import annotations

import json
import signal
import sys

from deepcoder_rlvr import (
    ExecutionTimeoutError,
    _execute_case,
    _guard_runtime,
)


def _raise_timeout(_signum, _frame):
    raise ExecutionTimeoutError("test timed out")


def main() -> None:
    payload = json.load(sys.stdin)
    test_cases = payload["test_cases"]
    case_timeouts = payload["case_timeouts"]
    fast_fail = bool(payload.get("fast_fail", True))

    try:
        _guard_runtime(
            payload.get("memory_limit_bytes"),
            payload["memory_limit_mode"],
            preload_modules=payload.get("preload_modules"),
        )
        signal.signal(signal.SIGALRM, _raise_timeout)
        results = []
        for test_case, timeout_seconds in zip(test_cases, case_timeouts, strict=True):
            signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
            try:
                result = _execute_case(payload["code"], test_case)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            results.append(result)
            if fast_fail and not result["passed"]:
                break
        output = {"test_results": results}
    except BaseException as exc:
        output = {"worker_error": f"{type(exc).__name__}: {exc}"}

    json.dump(output, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
