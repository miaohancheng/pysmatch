#!/usr/bin/env python3
"""Synchronize README files into docs markdown artifacts.

Usage:
  python scripts/sync_readme_to_docs.py
  python scripts/sync_readme_to_docs.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EN_REQUIRED = [
    "Installation",
    "Quickstart",
    "End-to-End Workflow",
    "Matching Strategies",
    "Evaluation",
    "Troubleshooting",
    "FAQ",
]

ZH_REQUIRED = [
    "安装",
    "快速开始",
    "端到端流程",
    "匹配策略",
    "评估",
    "故障排查",
    "常见问题",
]

EN_SECTION_FILES = {
    "Installation": "installation_en.md",
    "Quickstart": "quickstart_en.md",
    "End-to-End Workflow": "workflow_en.md",
    "Matching Strategies": "matching_en.md",
    "Evaluation": "evaluation_en.md",
    "Troubleshooting": "troubleshooting_en.md",
    "FAQ": "faq_en.md",
}

ZH_SECTION_FILES = {
    "安装": "installation_zh.md",
    "快速开始": "quickstart_zh.md",
    "端到端流程": "workflow_zh.md",
    "匹配策略": "matching_zh.md",
    "评估": "evaluation_zh.md",
    "故障排查": "troubleshooting_zh.md",
    "常见问题": "faq_zh.md",
}

H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # normalize trailing spaces for deterministic output
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip("\n") + "\n"
    return text


def parse_sections(readme_text: str, required: list[str], source_label: str) -> dict[str, str]:
    matches = list(H2_RE.finditer(readme_text))
    actual = [m.group(1).strip() for m in matches]

    if actual != required:
        missing = [h for h in required if h not in actual]
        unexpected = [h for h in actual if h not in required]
        details = [
            f"[{source_label}] H2 headings must exactly match required list in order.",
            f"Expected: {required}",
            f"Actual:   {actual}",
        ]
        if missing:
            details.append(f"Missing required headings: {missing}")
        if unexpected:
            details.append(f"Unexpected H2 headings: {unexpected}")
        raise ValueError("\n".join(details))

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(readme_text)
        block = readme_text[start:end].strip("\n") + "\n"
        sections[name] = block
    return sections


def with_generated_header(source_rel: str, body: str) -> str:
    header = (
        "<!-- AUTO-GENERATED: DO NOT EDIT. -->\n"
        f"<!-- Source: {source_rel} -->\n\n"
    )
    return normalize_newlines(header + body)


def section_to_page(section_markdown: str) -> str:
    # Convert the section heading from H2 to H1 for standalone page rendering.
    page = re.sub(r"^##\s+", "# ", section_markdown, count=1, flags=re.MULTILINE)
    # Shift nested headings up by one level (### -> ##, #### -> ###, ...).
    page = re.sub(
        r"^(#{3,6})(\s+)",
        lambda m: f"{m.group(1)[1:]}{m.group(2)}",
        page,
        flags=re.MULTILINE,
    )
    return page


def build_expected_outputs(root: Path) -> dict[Path, str]:
    readme_en_path = root / "README.md"
    readme_zh_path = root / "README_CHINESE.md"

    readme_en = normalize_newlines(readme_en_path.read_text(encoding="utf-8"))
    readme_zh = normalize_newlines(readme_zh_path.read_text(encoding="utf-8"))

    en_sections = parse_sections(readme_en, EN_REQUIRED, "README.md")
    zh_sections = parse_sections(readme_zh, ZH_REQUIRED, "README_CHINESE.md")

    out: dict[Path, str] = {}

    out[root / "docs" / "source" / "readme_en.md"] = with_generated_header(
        "../../README.md", readme_en
    )
    out[root / "docs" / "source" / "readme_zh.md"] = with_generated_header(
        "../../README_CHINESE.md", readme_zh
    )

    generated_dir = root / "docs" / "source" / "generated"
    for section, filename in EN_SECTION_FILES.items():
        out[generated_dir / filename] = with_generated_header(
            "../../../README.md",
            section_to_page(en_sections[section]),
        )

    for section, filename in ZH_SECTION_FILES.items():
        out[generated_dir / filename] = with_generated_header(
            "../../../README_CHINESE.md",
            section_to_page(zh_sections[section]),
        )

    return out


def file_content(path: Path) -> str:
    return normalize_newlines(path.read_text(encoding="utf-8"))


def run(check: bool) -> int:
    root = Path(__file__).resolve().parents[1]

    try:
        expected = build_expected_outputs(root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    mismatches: list[str] = []
    for path, content in expected.items():
        if not path.exists():
            mismatches.append(f"Missing generated file: {path}")
            continue
        actual = file_content(path)
        if actual != content:
            mismatches.append(f"Out-of-sync file: {path}")

    if check:
        if mismatches:
            print("README/docs sync check failed:", file=sys.stderr)
            for line in mismatches:
                print(f"  - {line}", file=sys.stderr)
            print(
                "Run `python scripts/sync_readme_to_docs.py` to regenerate artifacts.",
                file=sys.stderr,
            )
            return 1
        print("README/docs sync check passed.")
        return 0

    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print("Generated README mirror and section docs artifacts.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync README content to docs artifacts.")
    parser.add_argument("--check", action="store_true", help="Only check sync status without writing files.")
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
