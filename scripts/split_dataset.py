"""Split a paired slide dataset into train/val/test in a 7:2:1 ratio."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset_files import paired_samples, write_dataset
from utils.paths import project_path


def split_counts(number):
    if number < 3:
        raise ValueError("At least 3 slides are required for nonempty train, val and test sets.")
    weights = (7, 2, 1)
    counts = [number * weight // 10 for weight in weights]
    # Largest remainder allocation; ties follow train, val, test order.
    order = sorted(range(3), key=lambda i: (-(number * weights[i] % 10), i))
    for index in order[:number - sum(counts)]:
        counts[index] += 1
    # Small datasets still receive at least one slide per subset.
    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] += 1
    return dict(zip(("train", "val", "test"), counts))


def split_plan(sample_ids, seed):
    sample_ids = sorted(sample_ids)
    counts = split_counts(len(sample_ids))
    random.Random(seed).shuffle(sample_ids)
    samples = []
    start = 0
    for subset, count in counts.items():
        samples.extend({"id": sample_id, "subset": subset}
                       for sample_id in sorted(sample_ids[start:start + count]))
        start += count
    return samples, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Chosen dataset containing patches/ and st/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without creating dataset files")
    args = parser.parse_args(argv)
    root, output = project_path(args.root), project_path(args.output_dir)
    samples, counts = split_plan(paired_samples(root), args.seed)
    write_dataset(root, output, samples, mode=args.mode, dry_run=args.dry_run,
                  settings={"operation": "split_dataset", "split_unit": "slide", "seed": args.seed,
                            "ratios": {"train": 7, "val": 2, "test": 1}, "counts": counts})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
