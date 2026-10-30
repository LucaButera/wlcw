"""Remove experiment outputs, cached datasets, and downloaded model weights.

Run it from the repository root, within the `wlcw` environment (before removing it):
    python cleanup.py [-y]
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CacheNotFound

from env import CACHE

OUTPUTS = Path(__file__).resolve().parent / "outputs"

# Pretrained models downloaded from the HuggingFace Hub (Chronos2, Moirai2, TimesFM2.5)
HUB_MODELS = (
    "autogluon/chronos-2-small",
    "Salesforce/moirai-2.0-R-small",
    "google/timesfm-2.5-200m-pytorch",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-y", "--yes", action="store_true", help="do not ask for confirmation"
    )
    args = parser.parse_args()

    paths = [d for d in (OUTPUTS, CACHE) if d.exists()]
    try:
        # Locate the cached repos through the HuggingFace API, which resolves
        # HF_HOME/HF_HUB_CACHE, so that only these model folders are removed
        paths += [
            r.repo_path
            for r in scan_cache_dir().repos
            if r.repo_type == "model" and r.repo_id in HUB_MODELS
        ]
    except CacheNotFound:
        pass

    print("The following will be removed:")
    for path in paths:
        print(f"  {path}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    for path in paths:
        shutil.rmtree(path)
    print("Done.")


if __name__ == "__main__":
    main()
