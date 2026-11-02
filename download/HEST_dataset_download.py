import argparse
import csv
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "MahmoodLab/hest"


def load_hf_token():
    config_path = ROOT / "config" / "token.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f'Create {config_path} with an "hf_token" field.'
        )
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    token = config.get("hf_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Set hf_token in {config_path} before downloading.")
    return token.strip()


def download_hest(patterns, local_dir, *, revision=None, token=None):
    if not patterns:
        raise ValueError("No files selected for download.")
    return snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=local_dir,
        revision=revision,
        token=token if token is not None else load_hf_token(),
    )


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def latest_metadata(files):
    versions = []
    for name in files:
        match = re.fullmatch(r"HEST_v(\d+)_(\d+)_(\d+)\.csv", name)
        if match:
            versions.append((tuple(map(int, match.groups())), name))
    if not versions:
        raise ValueError("No versioned HEST metadata CSV found in the repository.")
    return max(versions)[1]


def read_metadata(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if "id" not in fields or not rows:
        raise ValueError("HEST metadata must contain an id column and sample rows.")
    return fields, rows


def parse_filter(value):
    field, separator, target = value.partition("=")
    if not separator or not field.strip() or not target.strip():
        raise argparse.ArgumentTypeError("Use COLUMN=VALUE, e.g. st_technology=Xenium")
    return field.strip(), target.strip()


def select_rows(rows, fields, filters):
    for field, _ in filters:
        if field not in fields:
            raise ValueError(f"Unknown column {field!r}. Available: {', '.join(fields)}")
    return [
        row for row in rows
        if all((row.get(field) or "").strip().casefold() == value.casefold()
               for field, value in filters)
    ]


def build_plan(rows, files):
    plan = []
    missing = []
    # Some metadata versions may repeat an id; download each slide only once.
    sample_ids = sorted({(row.get("id") or "").strip() for row in rows})
    for sample_id in sample_ids:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError(f"Invalid sample id in metadata: {sample_id!r}")
        for folder, suffix in (("patches", ".h5"), ("st", ".h5ad")):
            name = f"{folder}/{sample_id}{suffix}"
            if name not in files:
                missing.append(name)
                continue
            size = files[name].size
            if not isinstance(size, int) or size < 0:
                raise ValueError(f"No valid remote file size for {name}; cannot confirm total.")
            plan.append({"id": sample_id, "kind": folder, "path": name, "bytes": size})
    if missing:
        raise ValueError(
            "Selected slides have missing files; download stopped:\n  "
            + "\n  ".join(missing)
        )
    return plan


def format_size(size):
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024


def print_plan(plan):
    print(f"\n{'Slide':<20} {'patches':>14} {'st':>14} {'total':>14}")
    for patch, st in zip(plan[::2], plan[1::2]):
        print(f"{patch['id']:<20} {format_size(patch['bytes']):>14} "
              f"{format_size(st['bytes']):>14} "
              f"{format_size(patch['bytes'] + st['bytes']):>14}")
    for kind in ("patches", "st"):
        size = sum(item["bytes"] for item in plan if item["kind"] == kind)
        print(f"{kind}: {format_size(size)} ({size:,} bytes)")
    total = sum(item["bytes"] for item in plan)
    print(f"Slides: {len(plan) // 2}; files: {len(plan)}")
    print(f"TOTAL: {format_size(total)} ({total:,} bytes)")
    print("Full remote file sizes; excludes metadata, cache and transfer overhead.")
    print("Existing local files are included in this total; this is not remaining traffic.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select HEST slides by metadata and preview patches + st sizes. "
                    "Slide files are downloaded only with --download."
    )
    parser.add_argument("--organ", help="Organ, e.g. Breast, Lung or Brain")
    parser.add_argument("--filter", action="append", type=parse_filter, default=[],
                        metavar="COLUMN=VALUE", help="Repeat for AND filters; values ignore case")
    parser.add_argument("--list-values", metavar="COLUMN",
                        help="List metadata categories and counts, e.g. organ or st_technology")
    parser.add_argument("--local-dir", default="hest_data", help="Destination relative to project root")
    parser.add_argument("--revision", default="main", help="HF branch, tag or commit")
    parser.add_argument("--metadata-file", help="Metadata CSV in HF repository; default: latest HEST version")
    parser.add_argument("--report", help="Optional CSV report path relative to project root")
    parser.add_argument("--download", action="store_true", help="Download after printing the size summary")
    args = parser.parse_args(argv)
    if not args.organ and not args.filter and not args.list_values:
        parser.error("Choose --organ NAME, --filter COLUMN=VALUE, or --list-values COLUMN.")
    if args.list_values and args.download:
        parser.error("--list-values cannot be combined with --download.")
    return args


def main(argv=None):
    args = parse_args(argv)
    token = load_hf_token()
    print("Reading HEST metadata and remote file sizes...")
    info = HfApi(token=token).dataset_info(
        REPO_ID, revision=args.revision, files_metadata=True
    )
    if not info.sha:
        raise ValueError("Cannot determine HEST repository revision.")
    files = {item.rfilename: item for item in info.siblings}
    metadata_file = args.metadata_file or latest_metadata(files)
    if metadata_file not in files:
        raise ValueError(f"Metadata file not found: {metadata_file}")
    # Only this small CSV is fetched during preview, never patches or expression files.
    metadata_path = hf_hub_download(
        REPO_ID, metadata_file, repo_type="dataset", revision=info.sha, token=token
    )
    fields, rows = read_metadata(metadata_path)
    filters = list(args.filter)
    if args.organ:
        filters.append(("organ", args.organ.strip()))
    selected = select_rows(rows, fields, filters)
    print(f"Metadata: {metadata_file}; revision: {info.sha}")
    if args.list_values:
        if args.list_values not in fields:
            raise ValueError(f"Unknown column {args.list_values!r}. Available: {', '.join(fields)}")
        # Count unique slides rather than duplicated metadata rows.
        categories = {}
        for row in selected:
            key = (row.get(args.list_values) or "").strip() or "(empty)"
            categories.setdefault(key, set()).add(row["id"])
        for value, ids in sorted(categories.items()):
            print(f"{value}: {len(ids)} slides")
        return
    if not selected:
        raise ValueError("No slides match these filters. Use --list-values COLUMN to see categories.")
    plan = build_plan(selected, files)
    print("Filters: " + ", ".join(f"{key}={value}" for key, value in filters))
    print_plan(plan)
    local_dir = project_path(args.local_dir)
    print(f"Destination: {local_dir}")
    if args.report:
        report_path = project_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "kind", "path", "bytes", "revision"])
            writer.writeheader()
            writer.writerows(dict(item, revision=info.sha) for item in plan)
        print(f"Report: {report_path}")
    if not args.download:
        print("Preview only. Add --download to download these patches and st files.")
        print(f"To use this exact dataset revision, also pass --revision {info.sha}")
        return
    download_hest([item["path"] for item in plan], local_dir, revision=info.sha, token=token)
    print(f"Download complete: {local_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
