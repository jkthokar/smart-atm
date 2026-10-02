"""Split dataset/raw/<class>/ into dataset/train and dataset/test (default 80/20).

The test set is kept completely separate from training so the reported
accuracy is honest. If you collected images of several people, prefer
--by-person so the same person never appears in both sets (file names must
start with the person's name, e.g. `jay_001.jpg`).
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from smart_atm.config import CLASS_NAMES, ROOT

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=ROOT / "dataset" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--by-person", action="store_true", help="split by name prefix before '_'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for split in ("train", "test"):
        shutil.rmtree(args.out / split, ignore_errors=True)

    for cls in CLASS_NAMES:
        files = sorted(p for p in (args.raw / cls).glob("*") if p.suffix.lower() in IMG_EXT)
        if args.by_person:
            groups = defaultdict(list)
            for f in files:
                groups[f.stem.split("_")[0]].append(f)
            people = sorted(groups)
            rng.shuffle(people)
            n_test = max(1, round(len(people) * args.test_size))
            test = {f for p in people[:n_test] for f in groups[p]}
        else:
            shuffled = files[:]
            rng.shuffle(shuffled)
            test = set(shuffled[: round(len(files) * args.test_size)])

        for f in files:
            dest = args.out / ("test" if f in test else "train") / cls
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
        print(f"{cls:15s} train={len(files) - len(test):5d}  test={len(test):5d}")


if __name__ == "__main__":
    main()
