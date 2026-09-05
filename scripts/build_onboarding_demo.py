"""Build an offline synthetic onboarding model, not a real customer station."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.onboarding_demo import build_onboarding_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "projects" / "onboarding-demo")
    web = parser.add_mutually_exclusive_group()
    web.add_argument("--web-output", type=Path, default=ROOT / "web" / "public" / "demo")
    web.add_argument("--no-web", action="store_true", help="Build only the local project, without a viewer copy")
    args = parser.parse_args()
    try:
        report = build_onboarding_demo(args.output, None if args.no_web else args.web_output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Demo stopped safely: {error}\nNo files were deleted; choose a new version directory if needed.\n")


if __name__ == "__main__":
    main()
