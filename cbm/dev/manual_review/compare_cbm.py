#!/usr/bin/env python3
"""Generate a human-readable comparison of a CBM fork against original CBM.

Usage
-----
From the fork repository root:

    python dev/manual_review/compare_cbm.py \
        --fork . \
        --original ../cbm_python_original \
        --out dev/manual_review/output

The script does not import either package. It is therefore safe to use even
when their public APIs differ.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path
from typing import Iterable


CORE_FILES = [
    "cbm/__init__.py",
    "cbm/optimization.py",
    "cbm/map_estimation.py",
    "cbm/individual_fit.py",
    "cbm/model_selection.py",
    "cbm/group_bms.py",
    "cbm/hbi.py",
    "cbm/hbi_bound.py",
    "cbm/hbi_config.py",
    "cbm/hbi_exceedance.py",
    "cbm/hbi_logging.py",
    "cbm/hbi_types.py",
    "cbm/hbi_updates.py",
    "pyproject.toml",
    "README.md",
]

FEATURE_PATTERNS = {
    "GN optimization curvature": [
        "gauss_newton",
        "J.T @ J",
        "trial_func",
    ],
    "central-FD observed Hessian": [
        "central_fd",
        "_central_fd_hessian",
    ],
    "optional autodiff Hessian": [
        "autodiff",
        "jax.hessian",
    ],
    "Laplace validity separation": [
        "laplace_valid",
    ],
    "observed-Hessian diagnostics": [
        "hess_raw_min_eig",
        "hess_condition_number",
    ],
    "warning capture": [
        "catch_warnings",
        "warnings.warn",
    ],
    "display diagnostics": [
        "display",
        "search_path",
        "polish_path",
    ],
    "fixed parameters by zero variance": [
        "free_mask",
        "fixed_mask",
        "ParameterSpace",
    ],
    "latent tracking": [
        "evolution",
        "latent",
    ],
    "BMS between conditions/groups": [
        "group_bms_btw_conds",
        "group_bms_btw_groups",
    ],
}


def files_under(root: Path) -> set[str]:
    keep = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        keep.add(rel)
    return keep


def python_api(path: Path) -> dict:
    if not path.exists() or path.suffix != ".py":
        return {}

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__parse_error__": str(exc)}

    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            defaults = len(node.args.defaults)
            required = len(args) - defaults
            out[f"function:{node.name}"] = {
                "args": args,
                "required_args": args[:required],
            }
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            methods = [
                x.name
                for x in node.body
                if isinstance(x, ast.FunctionDef)
                and not x.name.startswith("_")
            ]
            out[f"class:{node.name}"] = {
                "methods": methods,
            }
    return out


def unified_diff(a: Path, b: Path, a_name: str, b_name: str) -> str:
    if not a.exists() and not b.exists():
        return ""
    a_lines = a.read_text(encoding="utf-8", errors="replace").splitlines(True) if a.exists() else []
    b_lines = b.read_text(encoding="utf-8", errors="replace").splitlines(True) if b.exists() else []
    return "".join(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=a_name,
            tofile=b_name,
            n=3,
        )
    )


def feature_matrix(fork: Path, original: Path) -> list[dict]:
    fork_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (fork / "cbm").glob("*.py")
        if p.is_file()
    )
    original_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (original / "cbm").glob("*.py")
        if p.is_file()
    )

    rows = []
    for feature, patterns in FEATURE_PATTERNS.items():
        rows.append({
            "feature": feature,
            "fork": any(p.lower() in fork_text.lower() for p in patterns),
            "original": any(p.lower() in original_text.lower() for p in patterns),
        })
    return rows


def hygiene(root: Path) -> list[str]:
    problems = []
    bad_names = {
        ".Rhistory",
        ".DS_Store",
    }

    for path in root.rglob("*"):
        if path.name in bad_names:
            problems.append(f"repository artifact: {path.relative_to(root)}")
        if path.is_dir() and path.name.endswith(".egg-info"):
            problems.append(f"generated package artifact: {path.relative_to(root)}")
        if path.is_file() and path.name.endswith("_legacy.py"):
            problems.append(f"legacy production file: {path.relative_to(root)}")

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="ignore")
        for token, label in [
            ('name = "cbm-local"', "package name is still cbm-local"),
            ('license = { text = "Proprietary" }', "license metadata says Proprietary"),
            ('https://example.com', "project URL is placeholder example.com"),
        ]:
            if token in text:
                problems.append(label)

    if not (root / "README.md").exists():
        problems.append("root README.md is missing")

    if not (root / "tests").exists():
        problems.append("no tests/ directory")

    return problems


def markdown_table(rows, headers):
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join([line1, line2] + body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fork = args.fork.resolve()
    original = args.original.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "diffs").mkdir(exist_ok=True)

    f_files = files_under(fork)
    o_files = files_under(original)

    inventory = {
        "fork_only": sorted(f_files - o_files),
        "original_only": sorted(o_files - f_files),
        "common": sorted(f_files & o_files),
    }
    (out / "inventory.json").write_text(
        json.dumps(inventory, indent=2),
        encoding="utf-8",
    )

    # Core diffs.
    for rel in CORE_FILES:
        diff = unified_diff(
            original / rel,
            fork / rel,
            f"original/{rel}",
            f"fork/{rel}",
        )
        safe = rel.replace("/", "__")
        (out / "diffs" / f"{safe}.diff").write_text(diff, encoding="utf-8")

    # API signatures.
    api = {}
    for rel in CORE_FILES:
        if not rel.endswith(".py"):
            continue
        api[rel] = {
            "original": python_api(original / rel),
            "fork": python_api(fork / rel),
        }
    (out / "public_api.json").write_text(
        json.dumps(api, indent=2),
        encoding="utf-8",
    )

    features = feature_matrix(fork, original)
    hygiene_issues = hygiene(fork)

    report = []
    report.append("# CBM fork vs original — manual review report\n")
    report.append("## Repository inventory\n")
    report.append(
        f"- Fork-only files: **{len(inventory['fork_only'])}**\n"
        f"- Original-only files: **{len(inventory['original_only'])}**\n"
        f"- Common files: **{len(inventory['common'])}**\n"
    )

    report.append("\n### Fork-only files\n")
    for rel in inventory["fork_only"]:
        report.append(f"- `{rel}`\n")

    report.append("\n### Original-only files\n")
    for rel in inventory["original_only"]:
        report.append(f"- `{rel}`\n")

    report.append("\n## Feature matrix\n")
    rows = []
    for row in features:
        rows.append([
            row["feature"],
            "yes" if row["original"] else "no",
            "yes" if row["fork"] else "no",
        ])
    report.append(
        markdown_table(
            rows,
            ["Feature", "Original", "Fork"],
        )
    )
    report.append("\n")

    report.append("\n## Pre-PR hygiene warnings\n")
    if hygiene_issues:
        for issue in hygiene_issues:
            report.append(f"- ⚠ {issue}\n")
    else:
        report.append("- none detected\n")

    report.append("\n## Public API review\n")
    report.append(
        "Inspect `public_api.json` for function/class signature changes. "
        "Every intentional breaking change should be explicitly mentioned "
        "in the pull-request description.\n"
    )

    report.append("\n## Line-by-line review\n")
    report.append(
        "Unified diffs for the core files are in `diffs/`. Recommended order:\n"
    )
    for rel in CORE_FILES:
        report.append(f"1. `{rel}`\n")

    (out / "REPORT.md").write_text("".join(report), encoding="utf-8")
    print(f"Review written to: {out}")
    print(f"Open first: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
