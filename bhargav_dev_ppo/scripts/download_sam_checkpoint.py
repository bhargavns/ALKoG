#!/usr/bin/env python3
"""Download the official SAM ViT-B checkpoint to the expected local path."""

import argparse
import os
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "sam_vit_b_01ec64.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size > 100_000_000:
        print(f"Checkpoint already exists: {args.output}")
        return
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    print(f"Downloading {args.url}")
    urllib.request.urlretrieve(args.url, temporary)
    if temporary.stat().st_size < 100_000_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded checkpoint is unexpectedly small")
    os.replace(temporary, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
