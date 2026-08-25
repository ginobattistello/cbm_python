#!/usr/bin/env python3
"""Compare a CBM fork against the original repository.

The ORIGINAL repository is the baseline.

The report answers four questions exhaustively:

1. Which files are NEW in the fork?
2. Which original files are DELETED in the fork?
3. Which files are SHARED by both repositories?
4. For every modified shared file, WHERE are the proposed edits?

Tracked files are obtained with ``git ls-files`` whenever possible, so the
inventory matches what belongs in source control rather than including local
generated/untracked artifacts.

Usage
-----
From the fork repository root:

    python cbm/dev/manual_review/compare_cbm.py \
        --fork . \
        --original ../cbm_python_original \
        --out cbm/dev/manual_review/output
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


TEXT_EXTENSIONS = {
    ".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".json",
    ".ini", ".cfg", ".sh", ".csv", ".rst", ".gitignore",
}


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str


def _git_tracked_files(root: Path) -> Optional[set[str]]:
    """Return git-tracked files, or None when root is not a git worktree."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    raw = proc.stdout.decode("utf-8", errors="surrogateescape")
    return {
        item
        for item in raw.split("\0")
        if item
    }


def _fallback_files(root: Path) -> set[str]:
    """Filesystem fallback used only when git metadata is unavailable."""
    out = set()

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)

        # Ignore local/generated material in fallback mode.
        if any(
            part in {
                ".git",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
            }
            for part in rel.parts
        ):
            continue

        if any(part.endswith(".egg-info") for part in rel.parts):
            continue

        out.add(rel.as_posix())

    return out


def tracked_files(root: Path) -> tuple[set[str], str]:
    files = _git_tracked_files(root)
    if files is not None:
        return files, "git ls-files"
    return _fallback_files(root), "filesystem fallback"


def sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def same_file(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False

    if a.stat().st_size != b.stat().st_size:
        return False

    return sha256(a) == sha256(b)


def looks_text(path: Path) -> bool:
    """Best-effort test for whether unified textual diff is appropriate."""
    if path.name == ".gitignore":
        return True

    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return False

    return b"\x00" not in sample


def read_text(path: Path) -> list[str]:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines(keepends=True)


def unified_diff(
    old_path: Path,
    new_path: Path,
    rel: str,
) -> str:
    return "".join(
        difflib.unified_diff(
            read_text(old_path),
            read_text(new_path),
            fromfile=f"original/{rel}",
            tofile=f"fork/{rel}",
            n=3,
        )
    )


_HUNK_RE = re.compile(
    r"^@@ -(?P<os>\d+)(?:,(?P<oc>\d+))? "
    r"\+(?P<ns>\d+)(?:,(?P<nc>\d+))? @@(?P<header>.*)$"
)


def parse_hunks(diff_text: str) -> list[Hunk]:
    hunks = []

    for line in diff_text.splitlines():
        match = _HUNK_RE.match(line)
        if match is None:
            continue

        hunks.append(
            Hunk(
                old_start=int(match.group("os")),
                old_count=int(match.group("oc") or "1"),
                new_start=int(match.group("ns")),
                new_count=int(match.group("nc") or "1"),
                header=match.group("header").strip(),
            )
        )

    return hunks


def _format_range(start: int, count: int) -> str:
    if count == 0:
        return "—"
    if count == 1:
        return str(start)
    return f"{start}-{start + count - 1}"


def _python_scopes(path: Path) -> list[tuple[int, int, str]]:
    """Return top-level class/function spans for a Python source file."""
    if path.suffix != ".py":
        return []

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return []

    scopes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            scopes.append(
                (node.lineno, end, f"function `{node.name}`")
            )

        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", node.lineno)
            scopes.append(
                (node.lineno, end, f"class `{node.name}`")
            )

            for child in node.body:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    child_end = getattr(
                        child,
                        "end_lineno",
                        child.lineno,
                    )
                    scopes.append(
                        (
                            child.lineno,
                            child_end,
                            f"method `{node.name}.{child.name}`",
                        )
                    )

    return scopes


def _affected_python_scopes(
    path: Path,
    hunks: list[Hunk],
) -> list[str]:
    scopes = _python_scopes(path)
    if not scopes:
        return []

    affected = []

    for start, end, label in scopes:
        for hunk in hunks:
            h_start = hunk.new_start
            h_end = (
                hunk.new_start + max(hunk.new_count, 1) - 1
            )

            if start <= h_end and end >= h_start:
                affected.append(label)
                break

    return sorted(set(affected))


def _possible_renames(
    original: Path,
    fork: Path,
    deleted: list[str],
    added: list[str],
) -> list[dict]:
    """Detect exact-content moves/renames among added/deleted files."""
    added_by_hash = {}

    for rel in added:
        digest = sha256(fork / rel)
        if digest is not None:
            added_by_hash.setdefault(digest, []).append(rel)

    pairs = []

    for old_rel in deleted:
        digest = sha256(original / old_rel)
        if digest in added_by_hash:
            for new_rel in added_by_hash[digest]:
                pairs.append({
                    "from": old_rel,
                    "to": new_rel,
                    "type": "exact-content rename/move candidate",
                })

    return pairs


def _table(headers, rows) -> str:
    if not rows:
        return "_None._\n"

    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    widths = [len(h) for h in headers]
    clean_rows = []

    for row in rows:
        vals = [clean(x) for x in row]
        clean_rows.append(vals)
        for i, value in enumerate(vals):
            widths[i] = max(widths[i], len(value))

    # Markdown does not need padded cells, but simple output is easier to diff.
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in clean_rows:
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    fork = args.fork.resolve()
    original = args.original.resolve()
    out = args.out.resolve()

    if not fork.exists():
        raise FileNotFoundError(f"Fork not found: {fork}")
    if not original.exists():
        raise FileNotFoundError(
            f"Original repository not found: {original}"
        )

    out.mkdir(parents=True, exist_ok=True)
    diff_dir = out / "diffs"
    diff_dir.mkdir(exist_ok=True)

    original_files, original_method = tracked_files(original)
    fork_files, fork_method = tracked_files(fork)

    # ORIGINAL is the baseline.
    added = sorted(fork_files - original_files)
    deleted = sorted(original_files - fork_files)
    shared = sorted(original_files & fork_files)

    shared_rows = []
    modified_files = []
    unchanged_files = []
    binary_modified = []

    details = {}

    for rel in shared:
        old_path = original / rel
        new_path = fork / rel

        if same_file(old_path, new_path):
            status = "unchanged"
            unchanged_files.append(rel)
            shared_rows.append([f"`{rel}`", status, "—"])
            continue

        status = "modified"
        modified_files.append(rel)

        if looks_text(old_path) and looks_text(new_path):
            diff_text = unified_diff(
                old_path,
                new_path,
                rel,
            )
            hunks = parse_hunks(diff_text)

            safe = (
                rel.replace("/", "__")
                .replace("\\", "__")
            )
            diff_path = diff_dir / f"{safe}.diff"
            diff_path.write_text(
                diff_text,
                encoding="utf-8",
            )

            scopes = _affected_python_scopes(
                new_path,
                hunks,
            )

            hunk_info = []
            for h in hunks:
                hunk_info.append({
                    "original_lines": _format_range(
                        h.old_start,
                        h.old_count,
                    ),
                    "fork_lines": _format_range(
                        h.new_start,
                        h.new_count,
                    ),
                    "context": h.header,
                })

            details[rel] = {
                "type": "text",
                "diff": str(
                    diff_path.relative_to(out)
                ),
                "hunks": hunk_info,
                "affected_python_scopes": scopes,
            }

            where = (
                ", ".join(
                    f"{h['original_lines']} → "
                    f"{h['fork_lines']}"
                    for h in hunk_info
                )
                if hunk_info
                else "text changed"
            )

            if scopes:
                where += "; " + ", ".join(scopes)

            shared_rows.append(
                [f"`{rel}`", status, where]
            )

        else:
            binary_modified.append(rel)
            details[rel] = {
                "type": "binary",
                "original_sha256": sha256(old_path),
                "fork_sha256": sha256(new_path),
            }
            shared_rows.append(
                [f"`{rel}`", "modified (binary)", "binary content differs"]
            )

    rename_candidates = _possible_renames(
        original,
        fork,
        deleted,
        added,
    )

    inventory = {
        "baseline": "original CBM",
        "inventory_method": {
            "original": original_method,
            "fork": fork_method,
        },
        "counts": {
            "original_tracked": len(original_files),
            "fork_tracked": len(fork_files),
            "added": len(added),
            "deleted": len(deleted),
            "shared": len(shared),
            "shared_modified": len(modified_files),
            "shared_unchanged": len(unchanged_files),
        },
        "added": added,
        "deleted": deleted,
        "shared": shared,
        "modified_shared": modified_files,
        "unchanged_shared": unchanged_files,
        "binary_modified": binary_modified,
        "possible_renames": rename_candidates,
        "modified_details": details,
    }

    (out / "inventory.json").write_text(
        json.dumps(
            inventory,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Separate machine-readable shared-file edit map.
    (out / "shared_file_edits.json").write_text(
        json.dumps(
            details,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = []

    report.append(
        "# Original CBM → proposed fork: exhaustive file review\n\n"
    )
    report.append(
        "The **original CBM repository is the baseline**. "
        "All categories below are computed over tracked files.\n\n"
    )

    report.append("## Summary\n\n")
    report.append(
        _table(
            ["Category", "Count"],
            [
                ["Original tracked files", len(original_files)],
                ["Fork tracked files", len(fork_files)],
                ["1. New files", len(added)],
                ["2. Deleted original files", len(deleted)],
                ["3. Shared files", len(shared)],
                ["4a. Shared + modified", len(modified_files)],
                ["4b. Shared + unchanged", len(unchanged_files)],
            ],
        )
    )

    report.append(
        f"\nInventory source: original = `{original_method}`; "
        f"fork = `{fork_method}`.\n"
    )

    # 1.
    report.append("\n## 1. New files added in the fork\n\n")
    if added:
        for rel in added:
            report.append(f"- `{rel}`\n")
    else:
        report.append("_None._\n")

    # 2.
    report.append(
        "\n## 2. Original CBM files deleted in the fork\n\n"
    )
    if deleted:
        for rel in deleted:
            report.append(f"- `{rel}`\n")
    else:
        report.append("_None._\n")

    if rename_candidates:
        report.append(
            "\n### Possible exact renames/moves\n\n"
        )
        report.append(
            "These deleted/new pairs have identical file contents and "
            "may represent moves rather than independent deletion/addition.\n\n"
        )
        for pair in rename_candidates:
            report.append(
                f"- `{pair['from']}` → `{pair['to']}`\n"
            )

    # 3.
    report.append("\n## 3. Shared files\n\n")
    report.append(
        _table(
            ["Shared file", "Status", "Where changed"],
            shared_rows,
        )
    )

    # 4.
    report.append(
        "\n## 4. Proposed edits within shared files\n\n"
    )

    if not modified_files:
        report.append("_No shared files are modified._\n")
    else:
        for rel in modified_files:
            info = details[rel]
            report.append(f"### `{rel}`\n\n")

            if info["type"] == "binary":
                report.append(
                    "Binary file differs; line-level diff is unavailable.\n\n"
                )
                continue

            scopes = info.get(
                "affected_python_scopes",
                [],
            )

            if scopes:
                report.append(
                    "**Affected Python scopes:** "
                    + ", ".join(scopes)
                    + "\n\n"
                )

            hunks = info.get("hunks", [])

            if hunks:
                report.append(
                    _table(
                        [
                            "Original line(s)",
                            "Fork line(s)",
                            "Diff context",
                        ],
                        [
                            [
                                h["original_lines"],
                                h["fork_lines"],
                                h["context"] or "—",
                            ]
                            for h in hunks
                        ],
                    )
                )
            else:
                report.append(
                    "_File changed, but no textual hunk metadata "
                    "was recovered._\n"
                )

            report.append(
                f"\nFull unified diff: `{info['diff']}`\n\n"
            )

    report.append(
        "\n## Review order\n\n"
        "For a PR review, inspect:\n\n"
        "1. **Deleted files** first — confirm every deletion is intentional.\n"
        "2. **New production files** — confirm each belongs upstream.\n"
        "3. **Modified shared files** — follow the line ranges above and "
        "open the corresponding unified diff.\n"
        "4. **Unchanged shared files** require no code review.\n"
    )

    (out / "REPORT.md").write_text(
        "".join(report),
        encoding="utf-8",
    )

    print("=" * 72)
    print("CBM repository comparison")
    print("=" * 72)
    print(f"Original tracked:      {len(original_files)}")
    print(f"Fork tracked:          {len(fork_files)}")
    print(f"New in fork:           {len(added)}")
    print(f"Deleted from original: {len(deleted)}")
    print(f"Shared:                {len(shared)}")
    print(f"  modified:            {len(modified_files)}")
    print(f"  unchanged:           {len(unchanged_files)}")
    print("-" * 72)
    print(f"Open: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
