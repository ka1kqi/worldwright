"""Tests for worldwright.utils.sandbox -- AST audit + restricted runtime."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from worldwright.utils import SandboxViolation, audit, compile_callable


# ---------- AST audit: rejects ----------

REJECTED_SOURCES = [
    ("import os\n", "os module"),
    ("import subprocess\nsubprocess.run('echo hi')\n", "subprocess"),
    ("from pathlib import Path\n", "pathlib"),
    ("from os import getcwd\n", "from-import os"),
    ("x = " + "ev" + "al('1+1')\n", "eval name"),
    ("__import__('os')\n", "dunder __import__ name"),
    ("open('/etc/passwd')\n", "open name"),
    ("type(x).__bases__\n", "dunder attribute"),
    ("from . import sibling\n", "relative import"),
]


@pytest.mark.parametrize("src,reason", REJECTED_SOURCES)
def test_audit_rejects_unsafe(src: str, reason: str) -> None:
    with pytest.raises(SandboxViolation):
        audit(src)


# ---------- AST audit: accepts ----------

def test_audit_accepts_allowed_imports() -> None:
    src = (
        "import numpy as np\n"
        "import math\n"
        "from worldwright.engine import WorldwrightScene\n"
        "def f(): return np.zeros(3) + math.pi\n"
    )
    audit(src)


def test_audit_rejects_syntax_error() -> None:
    with pytest.raises(SandboxViolation):
        audit("def f(:\n")


# ---------- Runtime: allowed code compiles + runs ----------

def test_compile_callable_returns_working_function() -> None:
    src = (
        "import math\n"
        "def success(state):\n"
        "    return math.sqrt(state.x * state.x) > 0.5\n"
    )
    fn = compile_callable(src, "success")
    assert fn(SimpleNamespace(x=1.0)) is True
    assert fn(SimpleNamespace(x=0.1)) is False


def test_compile_callable_with_injected_state() -> None:
    src = "def build(scene): scene.add_plane()\n"
    calls: list[str] = []

    class FakeScene:
        def add_plane(self) -> None:
            calls.append("plane")

    fn = compile_callable(src, "build")
    fn(FakeScene())
    assert calls == ["plane"]


def test_compile_callable_missing_function_raises() -> None:
    src = "def other(): return 1\n"
    with pytest.raises(KeyError):
        compile_callable(src, "missing")


def test_compile_callable_rejects_forbidden_runtime() -> None:
    with pytest.raises(SandboxViolation):
        compile_callable("def f(): open('/tmp/x')\n", "f")


def test_compile_callable_numpy_usage() -> None:
    src = (
        "import numpy as np\n"
        "def reward(state):\n"
        "    return float(np.linalg.norm(state.pos))\n"
    )
    fn = compile_callable(src, "reward")
    out = fn(SimpleNamespace(pos=np.array([3.0, 4.0, 0.0])))
    assert out == pytest.approx(5.0)
