#!/bin/python3
import sys, os
import urllib.request
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MIN_BYTES = 100_000_000  # a truncated download is far smaller than the real 375MB

required_arguments = []
optional_arguments = {
    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "out": os.path.join(_TROY_DEV, "models", "sam_vit_b_01ec64.pth"),
}

USAGE = """
download_sam_checkpoint.py -- fetch the SAM ViT-B checkpoint.

Downloads to models/sam_vit_b_01ec64.pth, which is where PerceptionPipeline
looks by default. Downloads to a .part file and renames on success, so an
interrupted run cannot leave a truncated checkpoint in place. Exits early if a
plausible checkpoint is already present.

    Optional:
        url=https://... out=.../models/sam_vit_b_01ec64.pth

    Example Usage:
        download_sam_checkpoint.py
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def download_checkpoint():
    url = g_ArgParse.get("url")
    out_path = g_ArgParse.get("out")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.isfile(out_path) and os.path.getsize(out_path) > _MIN_BYTES:
        print(f"Checkpoint already exists: {out_path}")
        return

    temporary = out_path + ".part"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, temporary)
    size = os.path.getsize(temporary)
    if size < _MIN_BYTES:
        os.remove(temporary)
        raise RuntimeError(f"Downloaded checkpoint is unexpectedly small ({size} bytes)")
    os.replace(temporary, out_path)
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main(inputArguments):
    initialize(inputArguments)
    download_checkpoint()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
