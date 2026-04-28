#!/usr/bin/env python3
"""Generate reproducible Python dependency lock files from requirements inputs.

This script resolves dependencies with pip's resolver (without installing)
and writes:
  - requirements.txt (runtime lock)
  - requirements-dev.txt (runtime lock + dev-only lock entries)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_INPUT = ROOT / "requirements.in"
DEV_INPUT = ROOT / "requirements-dev.in"
RUNTIME_LOCK = ROOT / "requirements.txt"
DEV_LOCK = ROOT / "requirements-dev.txt"


def canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_requirements(requirements_file: Path) -> dict[str, tuple[str, str]]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)

    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--dry-run",
            "--ignore-installed",
            "--report",
            str(report_path),
            "-r",
            str(requirements_file),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)

    resolved: dict[str, tuple[str, str]] = {}
    for entry in report.get("install", []):
        metadata = entry.get("metadata", {})
        name = metadata.get("name")
        version = metadata.get("version")
        if not name or not version:
            continue
        resolved[canonicalize(name)] = (name, version)
    return resolved


def write_runtime_lock(resolved: dict[str, tuple[str, str]]) -> None:
    lines = [
        "# Auto-generated lock file.",
        "# Source: requirements.in",
        "# Regenerate: python scripts/lock_requirements.py",
        "",
    ]
    for key in sorted(resolved):
        name, version = resolved[key]
        lines.append(f"{name}=={version}")
    RUNTIME_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dev_lock(
    runtime_resolved: dict[str, tuple[str, str]],
    dev_resolved: dict[str, tuple[str, str]],
) -> None:
    dev_only_keys = sorted(set(dev_resolved) - set(runtime_resolved))
    lines = [
        "# Auto-generated lock file.",
        "# Source: requirements-dev.in",
        "# Regenerate: python scripts/lock_requirements.py",
        "",
        "-r requirements.txt",
        "",
    ]
    for key in dev_only_keys:
        name, version = dev_resolved[key]
        lines.append(f"{name}=={version}")
    DEV_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    runtime_resolved = resolve_requirements(RUNTIME_INPUT)
    dev_resolved = resolve_requirements(DEV_INPUT)
    write_runtime_lock(runtime_resolved)
    write_dev_lock(runtime_resolved, dev_resolved)
    print("Generated requirements.txt and requirements-dev.txt from *.in inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
