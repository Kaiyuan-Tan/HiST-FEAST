"""Separate paired HEST slides into human/ and mouse/ using official metadata."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset_files import paired_samples, project_path, write_dataset


def load_metadata(local_file=None):
    if local_file:
        path = project_path(local_file)
        source = {"file": str(path)}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        from download.HEST_dataset_download import latest_metadata, load_hf_token
        token = load_hf_token()
        info = HfApi(token=token).dataset_info("MahmoodLab/hest")
        filename = latest_metadata(item.rfilename for item in info.siblings)
        path = Path(hf_hub_download(
            "MahmoodLab/hest", filename, repo_type="dataset", revision=info.sha, token=token
        ))
        source = {"repo": "MahmoodLab/hest", "file": filename, "revision": info.sha}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "species"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata CSV must contain id and species columns.")
        metadata = {}
        for row in reader:
            sample_id = row["id"].strip()
            species = row["species"].strip()
            if sample_id in metadata and metadata[sample_id]["species"].strip().casefold() != species.casefold():
                raise ValueError(f"Conflicting species entries for {sample_id}.")
            metadata[sample_id] = row
    return metadata, source


def species_plan(sample_ids, metadata):
    missing = sorted(set(sample_ids) - metadata.keys())
    if missing:
        raise ValueError("Slides absent from metadata: " + ", ".join(missing))
    counts = {"human": 0, "mouse": 0}
    samples = []
    for sample_id in sorted(sample_ids):
        row = metadata[sample_id]
        species = row["species"].strip()
        subset = {"homo sapiens": "human", "mus musculus": "mouse"}.get(species.casefold(), "other")
        counts[subset] = counts.get(subset, 0) + 1
        samples.append({
            "id": sample_id, "subset": subset, "species": species,
            "patient": row.get("patient", ""), "dataset_title": row.get("dataset_title", ""),
        })
    if counts.get("other"):
        print("Mixed, unknown or other species are kept in other/; inspect manifest.json.")
    return samples, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="hest_data", help="Input containing patches/ and st/")
    parser.add_argument("--output-dir", default="hest_data_species")
    parser.add_argument("--metadata", help="Local HEST CSV; default: fetch latest official metadata")
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without creating dataset files")
    args = parser.parse_args(argv)
    root, output = project_path(args.root), project_path(args.output_dir)
    sample_ids = paired_samples(root)
    metadata, source = load_metadata(args.metadata)
    samples, counts = species_plan(sample_ids, metadata)
    write_dataset(root, output, samples, mode=args.mode, dry_run=args.dry_run,
                  settings={"operation": "separate_species", "metadata": source, "counts": counts})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
