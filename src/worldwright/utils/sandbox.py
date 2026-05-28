"""Restricted-execution layer for LLM-generated scene + reward code.

Two-stage defence:

1.  **Static audit** (``audit``). Parse the source as an AST and walk it; reject
    imports outside the whitelist, dunder-attribute access, and a small set of
    obviously-unsafe builtin names.

2.  **Restricted runtime** (``compile_callable``). Run the audited source with a
    trimmed ``__builtins__`` dict so that even if a forbidden name slipped past
    the AST guard, it cannot be resolved at runtime.

Threat model is non-adversarial: this prevents typos and obvious mistakes
(a model that hallucinates ``import os`` and shells out to clean up). It is
NOT a sandbox against a motivated attacker; LLM-generated Python given full
``__builtins__`` access can always escape via creative reflection.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, Callable


# Top-level packages an LLM may import. Submodules are allowed implicitly
# (``worldwright.engine.handles`` matches ``worldwright.engine``).
ALLOWED_IMPORT_PREFIXES: frozenset[str] = frozenset({
    "numpy",
    "math",
    "worldwright.engine",
})

# Modules that are categorically forbidden, even if added to ALLOWED above.
DENIED_IMPORTS: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib", "io",
    "socket", "urllib", "http", "requests", "httpx",
    "asyncio", "threading", "multiprocessing",
    "ctypes", "importlib", "builtins", "pickle", "marshal",
    "tempfile", "glob",
})

# Builtin names that must never appear in source.
DENIED_NAMES: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__",
    "open", "input", "breakpoint", "help", "globals", "vars",
})

# Builtins exposed to sandboxed code.
_SAFE_BUILTINS_NAMES: frozenset[str] = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "Exception", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "IndexError", "AttributeError", "ZeroDivisionError",
    "True", "False", "None",
})


class SandboxViolation(Exception):
    """LLM-generated code violates the sandbox policy."""


def _import_allowed(name: str) -> bool:
    if name in DENIED_IMPORTS:
        return False
    root = name.split(".", 1)[0]
    if root in DENIED_IMPORTS:
        return False
    return any(
        name == p or name.startswith(p + ".")
        for p in ALLOWED_IMPORT_PREFIXES
    )


class _Auditor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _import_allowed(alias.name):
                raise SandboxViolation(
                    f"import {alias.name!r} not allowed "
                    f"(allowed prefixes: {sorted(ALLOWED_IMPORT_PREFIXES)})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None or node.level != 0:
            raise SandboxViolation("relative imports not allowed")
        if not _import_allowed(node.module):
            raise SandboxViolation(
                f"import from {node.module!r} not allowed "
                f"(allowed prefixes: {sorted(ALLOWED_IMPORT_PREFIXES)})"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in DENIED_NAMES:
            raise SandboxViolation(f"forbidden name: {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__") and len(node.attr) > 4:
            raise SandboxViolation(
                f"dunder attribute access forbidden: {node.attr!r}"
            )
        self.generic_visit(node)


def audit(source: str) -> None:
    """Raise SandboxViolation if ``source`` violates policy. Returns None on success."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxViolation(f"syntax error: {e}") from e
    _Auditor().visit(tree)


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Wrapped __import__ that re-checks the whitelist at runtime.

    The AST audit catches static ``import x`` statements; this catches anything
    that somehow reaches __import__ at runtime through a path we did not foresee.
    """
    if level != 0:
        raise SandboxViolation("relative imports not allowed at runtime")
    if not _import_allowed(name):
        raise SandboxViolation(
            f"runtime import {name!r} not allowed "
            f"(allowed prefixes: {sorted(ALLOWED_IMPORT_PREFIXES)})"
        )
    return getattr(builtins, "__import__")(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in _SAFE_BUILTINS_NAMES:
        if hasattr(builtins, n):
            out[n] = getattr(builtins, n)
    out["__import__"] = _safe_import
    return out


def compile_callable(
    source: str,
    fn_name: str,
    extra_globals: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Audit ``source``, run it in a restricted namespace, return ``fn_name``.

    ``extra_globals`` is merged into the execution globals -- used to inject
    pre-imported modules so the LLM source need not write its own imports.

    Raises:
        SandboxViolation: if the source fails the audit.
        KeyError: if ``fn_name`` is not defined after running the source.
    """
    audit(source)
    globals_dict: dict[str, Any] = {"__builtins__": _safe_builtins()}
    if extra_globals:
        globals_dict.update(extra_globals)
    _compile = getattr(builtins, "compile")
    _run = getattr(builtins, "exec")
    code_obj = _compile(source, "<sandbox>", "exec")
    _run(code_obj, globals_dict)
    if fn_name not in globals_dict:
        raise KeyError(
            f"function {fn_name!r} not defined in sandbox source "
            f"(defined: {sorted(k for k in globals_dict if not k.startswith('_'))})"
        )
    fn = globals_dict[fn_name]
    if not callable(fn):
        raise TypeError(f"{fn_name!r} is not callable (got {type(fn).__name__})")
    return fn
