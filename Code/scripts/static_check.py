"""Static repository validation that never imports project modules."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import yaml


def _module_exists(source_root: Path, name: str) -> bool:
    parts = name.split(".")
    return (source_root.joinpath(*parts).with_suffix(".py")).is_file() or (
        source_root.joinpath(*parts) / "__init__.py"
    ).is_file()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "src"
    errors: list[str] = []
    python_files = sorted([*source_root.rglob("*.py"), *(root / "scripts").glob("*.py")])
    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("drvcnet"):
                if not _module_exists(source_root, node.module):
                    errors.append(f"{path}:{node.lineno}: unresolved local module {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("drvcnet") and not _module_exists(source_root, alias.name):
                        errors.append(f"{path}:{node.lineno}: unresolved local module {alias.name}")

    for path in sorted((root / "configs").glob("*.yaml")):
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            errors.append(f"{path}: {error}")
    for path in sorted(root.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"{path}: {error}")

    summary = {
        "python_files": len(python_files),
        "yaml_files": len(list((root / "configs").glob("*.yaml"))),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
