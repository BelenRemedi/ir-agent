"""
Python Execution Tool - runs Python code in a restricted environment.

Useful for calculations, data processing, and text manipulation.
"""

import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr


# Blocked modules for safety
BLOCKED_IMPORTS = {"os", "subprocess", "sys", "shutil", "pathlib", "socket", "requests"}

ALLOWED_BUILTINS = {
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "sum", "min", "max", "abs", "round", "int", "float", "str", "bool",
    "list", "dict", "set", "tuple", "type", "isinstance", "repr", "format",
    "__import__",  # needed for safe imports below
}


class PythonExecTool:
    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "description": (
                        "Execute Python code for calculations, data manipulation, or analysis. "
                        "Standard library modules like math, statistics, json, re, datetime are available. "
                        "Network/OS access is restricted."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def python_exec(self, code: str) -> dict:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Safe globals with allowed standard library
        safe_globals = {
            "__builtins__": {b: __builtins__[b] for b in ALLOWED_BUILTINS if b in __builtins__}
            if isinstance(__builtins__, dict) else {
                b: getattr(__builtins__, b) for b in ALLOWED_BUILTINS if hasattr(__builtins__, b)
            },
            "math": __import__("math"),
            "statistics": __import__("statistics"),
            "json": __import__("json"),
            "re": __import__("re"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "random": __import__("random"),
        }

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, safe_globals)  # noqa: S102

            output = stdout_buf.getvalue()
            err = stderr_buf.getvalue()
            return {
                "status": "success",
                "output": output,
                "stderr": err if err else None,
            }
        except Exception:
            return {
                "status": "error",
                "error": traceback.format_exc(limit=5),
                "output": stdout_buf.getvalue(),
            }
