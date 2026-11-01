from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT_PER_TEST = float(os.getenv("DEEPCODER_RLVR_TIMEOUT_PER_TEST", "4"))
DEFAULT_MEMORY_LIMIT_BYTES = int(os.getenv("DEEPCODER_RLVR_MEMORY_LIMIT_BYTES", str(1024 * 1024 * 1024)))
DEFAULT_GLOBAL_TIMEOUT_SLACK = float(os.getenv("DEEPCODER_RLVR_GLOBAL_TIMEOUT_SLACK", "2"))
DEFAULT_OUTPUT_CAPTURE_CHARS = int(os.getenv("DEEPCODER_RLVR_OUTPUT_CAPTURE_CHARS", "1000000"))
DEFAULT_RESULT_OUTPUT_CHARS = int(os.getenv("DEEPCODER_RLVR_RESULT_OUTPUT_CHARS", "1000"))

MEMORY_LIMIT_MODE_NONE = "none"
MEMORY_LIMIT_MODE_FULL = "address_space_data_rss"
MEMORY_LIMIT_MODE_VIRTUAL = "virtual_memory"
DEFAULT_MEMORY_LIMIT_MODE = MEMORY_LIMIT_MODE_FULL
_SUPPORTED_MEMORY_LIMIT_MODES = {
    MEMORY_LIMIT_MODE_NONE,
    MEMORY_LIMIT_MODE_FULL,
    MEMORY_LIMIT_MODE_VIRTUAL,
}
_MISSING = object()

FORMAT_ERROR = "format_error"
SYNTAX_ERROR = "syntax_error"
RUNTIME_ERROR = "runtime_error"
WRONG_ANSWER = "wrong_answer"
TIMEOUT = "timeout"
TEST_CASE_ERROR = "test_case_error"

_CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


class ExecutionTimeoutError(TimeoutError):
    pass


class _BoundedTextBuffer(io.StringIO):
    """A text buffer that reports successful writes without retaining print bombs."""

    def __init__(self, max_chars: int):
        if max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        super().__init__()
        self.max_chars = int(max_chars)
        self.truncated = False
        self.discarded_non_whitespace = False
        self._captured_chars = 0

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"string argument expected, got {type(text).__name__}")

        remaining = max(self.max_chars - self._captured_chars, 0)
        retained = text[:remaining]
        if retained:
            super().write(retained)
            self._captured_chars += len(retained)

        discarded = text[remaining:]
        if discarded:
            self.truncated = True
            self.discarded_non_whitespace = self.discarded_non_whitespace or bool(discarded.strip())
        # TextIO.write() returns the accepted input length. Reporting the full
        # length keeps submitted programs from retrying discarded output.
        return len(text)


@dataclass
class CodeExtraction:
    code: str | None
    error: str | None = None
    error_message: str | None = None


def extract_single_python_code_block(model_output: str) -> CodeExtraction:
    """Extract exactly one Python markdown code block and validate syntax."""

    blocks = _CODE_BLOCK_RE.findall(model_output or "")
    if len(blocks) != 1:
        return CodeExtraction(
            code=None,
            error=FORMAT_ERROR,
            error_message=f"expected exactly one markdown code block, found {len(blocks)}",
        )

    language, code = blocks[0]
    language = language.strip().lower()
    if language not in {"", "py", "python", "python3"}:
        return CodeExtraction(
            code=None,
            error=FORMAT_ERROR,
            error_message=f"expected a Python code block, got {language!r}",
        )

    code = code.strip()
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return CodeExtraction(code=None, error=SYNTAX_ERROR, error_message=str(exc))

    return CodeExtraction(code=code)


def _safe_repr(value: Any, max_chars: int = 1000) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def _safe_text(value: Any, max_chars: int = DEFAULT_RESULT_OUTPUT_CHARS) -> str:
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def _captured_output_text(value: Any, *, truncated: bool) -> str:
    text = str(value).strip()
    if not truncated:
        return _safe_text(text)

    suffix = "... [truncated]"
    prefix_chars = max(DEFAULT_RESULT_OUTPUT_CHARS - len(suffix), 0)
    return text[:prefix_chars] + suffix


def _output_capture_limit(expected: Any) -> int:
    if isinstance(expected, list):
        expected_text = "\n".join(str(item) for item in expected)
    else:
        expected_text = str(expected)
    # Keep enough bytes to compare any trusted expected answer exactly. The
    # extra character proves that non-whitespace output beyond it is wrong.
    return max(DEFAULT_OUTPUT_CAPTURE_CHARS, len(expected_text) + 1)


def _set_memory_limit(memory_limit_bytes: int | None, memory_limit_mode: str) -> None:
    if not memory_limit_bytes or memory_limit_mode == MEMORY_LIMIT_MODE_NONE:
        return
    if memory_limit_mode not in _SUPPORTED_MEMORY_LIMIT_MODES:
        raise ValueError(f"unsupported memory limit mode: {memory_limit_mode!r}")
    import resource

    limit_names = (
        ("RLIMIT_AS",)
        if memory_limit_mode == MEMORY_LIMIT_MODE_VIRTUAL
        else ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS")
    )
    applied = []
    for name in limit_names:
        if hasattr(resource, name):
            limit = getattr(resource, name)
            resource.setrlimit(limit, (memory_limit_bytes, memory_limit_bytes))
            applied.append(name)
    if not applied:
        raise RuntimeError(f"no supported OS memory limits for mode {memory_limit_mode!r}")


def _blocked(*args, **kwargs):
    raise PermissionError("operation blocked by DeepCoder RLVR execution guard")


def _guard_runtime(
    memory_limit_bytes: int | None,
    memory_limit_mode: str,
    preload_modules: list[str] | None = None,
) -> None:
    _set_memory_limit(memory_limit_bytes, memory_limit_mode)

    with contextlib.suppress(Exception):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    # Some trusted hidden-test harnesses (MBPP+) need NumPy. Import requested
    # modules after applying RLIMIT but before filesystem/threading guards, so
    # their normal initialization is counted against the submitted-program cap.
    for module_name in preload_modules or []:
        __import__(module_name)

    for name in (
        "system",
        "popen",
        "remove",
        "removedirs",
        "rmdir",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "rename",
        "renames",
        "replace",
        "unlink",
        "chmod",
        "chown",
        "chroot",
        "fchmod",
        "fchown",
        "truncate",
        "putenv",
    ):
        if hasattr(os, name):
            with contextlib.suppress(Exception):
                setattr(os, name, _blocked)

    with contextlib.suppress(Exception):
        import shutil

        shutil.rmtree = _blocked
        shutil.move = _blocked
        shutil.chown = _blocked

    with contextlib.suppress(Exception):
        import subprocess

        subprocess.Popen = _blocked
        subprocess.call = _blocked
        subprocess.run = _blocked
        subprocess.check_call = _blocked
        subprocess.check_output = _blocked

    for name in (
        "socket",
        "ssl",
        "urllib",
        "urllib3",
        "requests",
        "http",
        "ftplib",
        "telnetlib",
        "multiprocessing",
        "threading",
        "ctypes",
        "resource",
        "signal",
        "psutil",
    ):
        sys.modules[name] = None

    builtins.open = _blocked
    builtins.exit = _blocked
    builtins.quit = _blocked


def _restricted_builtins() -> dict[str, Any]:
    allowed_names = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "IndexError",
        "KeyError",
        "RuntimeError",
        "StopIteration",
        "NotImplementedError",
    }
    safe = {name: getattr(builtins, name) for name in allowed_names if hasattr(builtins, name)}

    allowed_modules = {
        "array",
        "bisect",
        "collections",
        "copy",
        "cmath",
        "dataclasses",
        "datetime",
        "decimal",
        "functools",
        "fractions",
        "heapq",
        "itertools",
        "json",
        "math",
        "numpy",
        "operator",
        "queue",
        "random",
        "re",
        "statistics",
        "string",
        "sys",
        "typing",
    }
    real_import = builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed_modules:
            raise ImportError(f"import of module {name!r} is blocked")
        return real_import(name, globals, locals, fromlist, level)

    safe["__import__"] = restricted_import
    safe["__build_class__"] = builtins.__build_class__
    safe["input"] = builtins.input
    safe["open"] = _blocked
    return safe


def _execution_namespace() -> dict[str, Any]:
    namespace = {
        "__builtins__": _restricted_builtins(),
        "__name__": "__not_main__",
    }
    try:
        import typing

        for name in (
            "Any",
            "Callable",
            "Deque",
            "Dict",
            "Iterable",
            "Iterator",
            "List",
            "Mapping",
            "Optional",
            "Sequence",
            "Set",
            "Tuple",
            "Union",
        ):
            if hasattr(typing, name):
                namespace[name] = getattr(typing, name)
    except Exception:
        pass
    return namespace


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    raise TypeError(f"non-jsonable value {type(value).__name__}")


def _loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _normalize_output_text(text: Any) -> list[str]:
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    return str(text).strip().split()


def _outputs_match(actual: Any, expected: Any, functional: bool) -> bool:
    if functional:
        expected_value = _loads_maybe(expected)
        actual_value = _jsonable(actual)
        expected_value = _jsonable(expected_value)
        if actual_value == expected_value:
            return True
        if (
            isinstance(expected_value, list)
            and len(expected_value) == 1
            and actual_value == expected_value[0]
        ):
            return True
        return False

    actual_text = actual if isinstance(actual, str) else str(actual)
    expected_text = "\n".join(str(x) for x in expected) if isinstance(expected, list) else str(expected)
    if actual_text.strip() == expected_text.strip():
        return True
    return _normalize_output_text(actual_text) == _normalize_output_text(expected_text)


def _coerce_input_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def _parse_function_args(raw_input: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(raw_input, dict):
        return [], raw_input
    if isinstance(raw_input, tuple):
        return list(raw_input), {}
    if isinstance(raw_input, list):
        return raw_input, {}
    if raw_input is None:
        return [], {}

    text = str(raw_input).strip()
    if text == "":
        return [], {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        args = []
        for line in lines:
            args.append(_loads_maybe(line))
        return args, {}

    try:
        return [json.loads(text)], {}
    except Exception:
        pass

    pieces = text.split()
    if len(pieces) > 1:
        args = []
        for piece in pieces:
            args.append(_loads_maybe(piece))
        return args, {}

    return [text], {}


def _find_callable(namespace: dict[str, Any], fn_name: str) -> Any:
    if fn_name and callable(namespace.get(fn_name)):
        return namespace[fn_name]
    solution_cls = namespace.get("Solution")
    if solution_cls is not None:
        instance = solution_cls()
        if fn_name and callable(getattr(instance, fn_name, None)):
            return getattr(instance, fn_name)
    if not fn_name:
        funcs = [value for value in namespace.values() if callable(value)]
        user_funcs = [f for f in funcs if getattr(f, "__module__", None) in {None, "__not_main__"}]
        if user_funcs:
            return user_funcs[0]
    raise AttributeError(f"function {fn_name!r} not found")


def _case_fn_name(test_case: dict[str, Any], top_level_metadata: dict[str, Any] | None = None) -> str:
    metadata = test_case.get("metadata") or {}
    top_level_metadata = top_level_metadata or {}
    return (
        test_case.get("fn_name")
        or test_case.get("func_name")
        or metadata.get("fn_name")
        or metadata.get("func_name")
        or top_level_metadata.get("fn_name")
        or top_level_metadata.get("func_name")
        or ""
    )


def _case_testtype(test_case: dict[str, Any]) -> str:
    metadata = test_case.get("metadata") or {}
    return str(test_case.get("testtype") or metadata.get("testtype") or "").strip().lower()


def _case_policy_value(test_case: dict[str, Any], key: str) -> Any:
    if key in test_case:
        return test_case[key]
    metadata = test_case.get("metadata") or {}
    return metadata[key] if key in metadata else _MISSING


def _case_timeout_seconds(test_case: dict[str, Any], default: float) -> float:
    value = _case_policy_value(test_case, "timeout_seconds")
    if value is _MISSING:
        value = _case_policy_value(test_case, "time_limit")
    timeout = default if value is _MISSING or value is None else float(value)
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    return timeout


def _case_memory_limit(
    test_case: dict[str, Any],
    default_bytes: int | None,
    default_mode: str,
) -> tuple[int | None, str]:
    raw_bytes = _case_policy_value(test_case, "memory_limit_bytes")
    raw_mode = _case_policy_value(test_case, "memory_limit_mode")

    if raw_mode is _MISSING:
        if raw_bytes is None:
            return None, MEMORY_LIMIT_MODE_NONE
        mode = default_mode
    else:
        mode = str(raw_mode).strip().lower()
    if mode not in _SUPPORTED_MEMORY_LIMIT_MODES:
        raise ValueError(f"unsupported memory limit mode: {mode!r}")
    if mode == MEMORY_LIMIT_MODE_NONE:
        return None, mode

    value = default_bytes if raw_bytes is _MISSING else raw_bytes
    if value is None:
        raise ValueError(f"memory_limit_bytes is required for mode {mode!r}")
    limit_bytes = int(value)
    if limit_bytes <= 0:
        raise ValueError(f"memory_limit_bytes must be positive, got {limit_bytes}")
    return limit_bytes, mode


def _case_preload_modules(test_case: dict[str, Any]) -> list[str]:
    raw_modules = _case_policy_value(test_case, "preload_modules")
    if raw_modules is _MISSING or raw_modules is None:
        return []
    if not isinstance(raw_modules, list) or not all(isinstance(name, str) and name for name in raw_modules):
        raise ValueError("preload_modules must be a list of non-empty module names")
    unsupported = sorted(set(raw_modules) - {"numpy"})
    if unsupported:
        raise ValueError(f"unsupported preload modules: {unsupported}")
    return raw_modules


def _normalize_tests(test_data: Any) -> list[dict[str, Any]]:
    if isinstance(test_data, str):
        test_data = json.loads(test_data)

    top_level_metadata: dict[str, Any] = {}
    if isinstance(test_data, dict):
        top_level_metadata = test_data.get("metadata") or {}
        if "inputs" in test_data and "outputs" in test_data:
            inputs = test_data["inputs"]
            outputs = test_data["outputs"]
            if len(inputs) != len(outputs):
                raise ValueError("inputs and outputs have different lengths")
            metadata = dict(top_level_metadata)
            for key in ("fn_name", "func_name"):
                if test_data.get(key) and key not in metadata:
                    metadata[key] = test_data[key]
            return [
                {"input": inp, "output": out, "metadata": dict(metadata)}
                for inp, out in zip(inputs, outputs, strict=True)
            ]
        if "input" in test_data and "output" in test_data:
            return [test_data]
        if "test_cases" in test_data and isinstance(test_data["test_cases"], list):
            return _normalize_tests(test_data["test_cases"])
        raise ValueError("unsupported test dictionary format")

    if isinstance(test_data, list):
        cases = []
        for item in test_data:
            if not isinstance(item, dict):
                raise ValueError("test case must be a dictionary")
            if "input" not in item or "output" not in item:
                raise ValueError("test case must contain input and output")
            cases.append(item)
        return cases

    raise ValueError(f"unsupported tests type {type(test_data).__name__}")


# The worker imports this file as a sibling module before applying its runtime guard.
try:
    from .code_eval_checkers import check_output as _check_eval_output
except ImportError:
    import importlib.util as _checker_importlib

    _checker_spec = _checker_importlib.spec_from_file_location(
        "code_eval_checkers", os.path.join(os.path.dirname(__file__), "code_eval_checkers.py")
    )
    _checker_module = _checker_importlib.module_from_spec(_checker_spec)
    _checker_spec.loader.exec_module(_checker_module)
    _check_eval_output = _checker_module.check_output


def _stdin_output_matches(actual, test_case):
    policy = (test_case.get("metadata") or {}).get("output_checker")
    if policy is not None:
        return _check_eval_output(actual, test_case.get("output"), test_case.get("input"), policy)
    return _outputs_match(actual, test_case.get("output"), functional=False)


def _execute_case(code: str, test_case: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    fn_name = _case_fn_name(test_case)
    functional = bool(fn_name)
    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    stdout_buffer = _BoundedTextBuffer(_output_capture_limit(test_case.get("output")))
    stderr_buffer = _BoundedTextBuffer(DEFAULT_OUTPUT_CAPTURE_CHARS)

    try:
        namespace = _execution_namespace()
        local_eval = bool((test_case.get("metadata") or {}).get("local_code_eval"))
        if local_eval:
            namespace["__builtins__"]["type"] = builtins.type

            def eval_open(file, mode="r", *args, **kwargs):
                if file != 0 or mode not in ("r", "rt", "rb"):
                    return _blocked()
                text = _coerce_input_text(test_case.get("input"))
                return io.BytesIO(text.encode()) if mode == "rb" else io.StringIO(text)

            namespace["__builtins__"]["open"] = eval_open
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        testtype = _case_testtype(test_case)
        if testtype == "code":
            namespace["__name__"] = "__not_main__"
            exec(compile(code, "solution.py", "exec"), namespace, namespace)
            exec(
                compile(_coerce_input_text(test_case.get("input")), "tests.py", "exec"), namespace, namespace
            )
            actual = "All tests pass"
            passed = True
        elif functional:
            exec(compile(code, "solution.py", "exec"), namespace, namespace)
            func = _find_callable(namespace, fn_name)
            args, kwargs = _parse_function_args(test_case.get("input"))
            actual = func(**kwargs) if kwargs else func(*args)
            passed = _outputs_match(actual, test_case.get("output"), functional=functional)
        else:
            namespace["__name__"] = "__main__"
            input_text = _coerce_input_text(test_case.get("input"))
            sys.stdin = (
                io.TextIOWrapper(io.BytesIO(input_text.encode())) if local_eval else io.StringIO(input_text)
            )
            sys.stdout = stdout_buffer
            try:
                exec(compile(code, "solution.py", "exec"), namespace, namespace)
            except SystemExit as exc:
                if not local_eval or exc.code not in (None, 0):
                    raise
            actual = stdout_buffer.getvalue()
            passed = not stdout_buffer.discarded_non_whitespace and _stdin_output_matches(actual, test_case)

        result = {
            "passed": passed,
            "actual": (
                _safe_repr(actual)
                if functional
                else _captured_output_text(actual, truncated=stdout_buffer.truncated)
            ),
            "expected": _safe_repr(test_case.get("output")),
            "error": None if passed else WRONG_ANSWER,
            "time": time.time() - start,
        }
    except ExecutionTimeoutError:
        result = _timeout_result(test_case)
    except BaseException as exc:
        result = {
            "passed": False,
            "actual": "",
            "expected": _safe_repr(test_case.get("output")),
            "error": RUNTIME_ERROR,
            "error_message": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "time": time.time() - start,
        }
    finally:
        sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr
    return result


def _timeout_result(test_case: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": False,
        "actual": "",
        "expected": _safe_repr(test_case.get("output")),
        "error": TIMEOUT,
        "error_message": "test timed out",
        "time": float("inf"),
    }


def _worker_script_path() -> str:
    return os.path.join(os.path.dirname(__file__), "deepcoder_rlvr_worker.py")


def run_execution_tests(
    code: str,
    tests: str | list[dict[str, Any]] | dict[str, Any],
    *,
    timeout_per_test: float = DEFAULT_TIMEOUT_PER_TEST,
    global_timeout: float | None = None,
    memory_limit_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES,
    memory_limit_mode: str = DEFAULT_MEMORY_LIMIT_MODE,
    fast_fail: bool = True,
) -> dict[str, Any]:
    try:
        test_cases = _normalize_tests(tests)
        case_timeouts = [_case_timeout_seconds(test_case, timeout_per_test) for test_case in test_cases]
        case_memory_limits = [
            _case_memory_limit(test_case, memory_limit_bytes, memory_limit_mode) for test_case in test_cases
        ]
        preload_modules = sorted(
            {module_name for test_case in test_cases for module_name in _case_preload_modules(test_case)}
        )
    except Exception as exc:
        return {
            "all_passed": False,
            "total_tests": 0,
            "passed_tests": 0,
            "test_results": [],
            "error": TEST_CASE_ERROR,
            "error_message": str(exc),
        }

    if not test_cases:
        return {
            "all_passed": False,
            "total_tests": 0,
            "passed_tests": 0,
            "test_results": [],
            "error": TEST_CASE_ERROR,
            "error_message": "no tests provided",
        }

    total_tests = len(test_cases)
    if global_timeout is None:
        global_timeout = sum(case_timeouts) + DEFAULT_GLOBAL_TIMEOUT_SLACK

    if len(set(case_memory_limits)) != 1:
        return {
            "all_passed": False,
            "total_tests": 0,
            "passed_tests": 0,
            "test_results": [],
            "error": TEST_CASE_ERROR,
            "error_message": "all tests in one problem must use the same memory policy",
        }

    worker_payload = {
        "code": code,
        "test_cases": test_cases,
        "case_timeouts": case_timeouts,
        "memory_limit_bytes": case_memory_limits[0][0],
        "memory_limit_mode": case_memory_limits[0][1],
        "preload_modules": preload_modules,
        "fast_fail": fast_fail,
    }
    try:
        completed = subprocess.run(
            [sys.executable, _worker_script_path()],
            input=json.dumps(worker_payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=global_timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip() or f"execution worker exited with code {completed.returncode}"
            )
        worker_result = json.loads(completed.stdout)
        if worker_result.get("worker_error"):
            raise RuntimeError(str(worker_result["worker_error"]))
        results = worker_result["test_results"]
    except subprocess.TimeoutExpired:
        results = [_timeout_result(test_cases[0])]
    except Exception as exc:
        results = [
            {
                "passed": False,
                "actual": "",
                "expected": _safe_repr(test_cases[0].get("output")),
                "error": RUNTIME_ERROR,
                "error_message": f"execution worker failed: {exc}",
                "time": float("inf"),
            }
        ]

    for test_idx, result in enumerate(results):
        test_case = test_cases[test_idx]
        result["test_idx"] = test_idx
        result["input"] = test_case.get("input")
        result["timeout_seconds"] = case_timeouts[test_idx]
        result["memory_limit_bytes"], result["memory_limit_mode"] = case_memory_limits[test_idx]

    passed_tests = sum(1 for result in results if result.get("passed"))
    all_passed = passed_tests == total_tests and len(results) == total_tests
    first_error = next((r for r in results if not r.get("passed")), {})
    metadata = {
        "all_passed": all_passed,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "test_results": results,
    }
    if not all_passed:
        metadata["error"] = first_error.get("error") or WRONG_ANSWER
        if first_error.get("error_message"):
            metadata["error_message"] = first_error["error_message"]
    return metadata


def _format_failure_result(error: str, error_message: str = "") -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "passed_tests_ratio": 0.0,
        "all_passed": False,
        "total_tests": 0,
        "passed_tests": 0,
        "test_results": [],
        "incorrect_format": 1 if error == FORMAT_ERROR else 0,
        "timed_out": 0,
        "error": error,
        "error_message": error_message,
        "format_reward": 0.0 if error == FORMAT_ERROR else 1.0,
    }


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    timeout_per_test: float = DEFAULT_TIMEOUT_PER_TEST,
    global_timeout: float | None = None,
    memory_limit_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES,
    memory_limit_mode: str = DEFAULT_MEMORY_LIMIT_MODE,
    fast_fail: bool = True,
) -> dict[str, Any]:
    extraction = extract_single_python_code_block(solution_str)
    if extraction.code is None:
        return _format_failure_result(extraction.error or FORMAT_ERROR, extraction.error_message or "")

    metadata = run_execution_tests(
        extraction.code,
        ground_truth,
        timeout_per_test=timeout_per_test,
        global_timeout=global_timeout,
        memory_limit_bytes=memory_limit_bytes,
        memory_limit_mode=memory_limit_mode,
        fast_fail=fast_fail,
    )
    total_tests = metadata["total_tests"]
    passed_tests = metadata["passed_tests"]
    ratio = float(passed_tests) / float(total_tests) if total_tests else 0.0
    reward = 1.0 if metadata["all_passed"] else 0.0

    return {
        "score": reward,
        # Problem-level exact correctness: partial hidden-test passes are kept
        # separately as passed_tests_ratio and never count as a correct sample.
        "acc": reward,
        "passed_tests_ratio": ratio,
        "all_passed": metadata["all_passed"],
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "test_results": metadata["test_results"],
        "incorrect_format": 0,
        "timed_out": 1 if metadata.get("error") == TIMEOUT else 0,
        "error": metadata.get("error"),
        "error_message": metadata.get("error_message", ""),
        "format_reward": 1.0,
    }


def compute_score_for_source(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    return compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
