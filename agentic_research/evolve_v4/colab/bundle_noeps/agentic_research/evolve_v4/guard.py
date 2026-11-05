"""Static guard for LLM-written policies: blocks file, process and introspection access.

This catches mistakes and casual shortcuts by a cooperative model; it is not an
OS security boundary.
"""
import ast

ALLOWED_IMPORTS = {"numpy", "math", "itertools", "functools", "heapq", "collections",
                   "agentic_research.evolve_v4.helpers"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview", "help"}
BANNED_ATTRS = {"f_back", "f_globals", "f_locals", "gi_frame", "tb_frame", "__dict__",
                "__class__", "__subclasses__", "__globals__", "__code__", "__builtins__",
                "revealed", "epsilon", "load", "loadtxt", "fromfile", "genfromtxt", "memmap", "save", "savez", "tofile"}


def check(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name not in ALLOWED_IMPORTS:
                raise ValueError(f"import not allowed: {name}")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ValueError(f"name not allowed: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            raise ValueError(f"attribute not allowed: {node.attr}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "choose" for n in tree.body):
        raise ValueError("program must define choose(view)")
