from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.web_preview_acceptance import accept_web_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Accept a local final-model web preview")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--preview-directory", required=True, type=Path)
    parser.add_argument("--final-qa", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--browser-visual-review-completed", action="store_true")
    args = parser.parse_args()
    report = accept_web_preview(
        base_url=args.base_url,
        preview_directory=args.preview_directory,
        final_qa_path=args.final_qa,
        output_path=args.output,
        browser_visual_review_completed=args.browser_visual_review_completed,
    )
    print(f"technical_passed={report['technical_passed']}")
    print(f"browser_visual_review_completed={report['browser_visual_review_completed']}")
    print(f"status={report['status']}")


if __name__ == "__main__":
    main()
