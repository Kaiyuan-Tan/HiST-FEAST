"""Shared file operations for preparing paired HEST slide directories."""
import json
import os
import shutil
from collections import Counter


def paired_samples(root):
    patches = {path.stem for path in (root / "patches").glob("*.h5") if path.is_file()}
    expression = {path.stem for path in (root / "st").glob("*.h5ad") if path.is_file()}
    if not patches and not expression:
        raise ValueError(f"No slide files found in {root}/patches and {root}/st.")
    if patches != expression:
        raise ValueError(
            "Unpaired slides. Complete the input dataset first.\n"
            f"Missing st files: {', '.join(sorted(patches - expression)) or '(none)'}\n"
            f"Missing patches: {', '.join(sorted(expression - patches)) or '(none)'}"
        )
    return sorted(patches)


def write_dataset(source, output, samples, *, mode, settings, dry_run=False):
    """Create subset/patches + subset/st without changing source file contents."""
    if mode not in ("hardlink", "copy"):
        raise ValueError(f"Unknown transfer mode: {mode}")
    if output == source or output in source.parents:
        raise ValueError("Output must not be the input directory or one of its parents.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be absent or empty: {output}")
    ids = [sample["id"] for sample in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Each input slide must occur exactly once in the output plan.")
    groups = settings["counts"]
    for sample in samples:
        if sample["subset"] not in groups:
            raise ValueError("Unknown output subset.")
        for folder, suffix in (("patches", ".h5"), ("st", ".h5ad")):
            if not (source / folder / f"{sample['id']}{suffix}").is_file():
                raise ValueError(f"Missing input file for {sample['id']}: {folder}")
    if dict(Counter(sample["subset"] for sample in samples)) != {
        key: value for key, value in groups.items() if value
    }:
        raise ValueError("Subset counts do not match the output plan.")

    if mode == "hardlink":
        existing_parent = output
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        device = existing_parent.stat().st_dev
        if any((source / folder / f"{sample['id']}{suffix}").stat().st_dev != device
               for sample in samples for folder, suffix in (("patches", ".h5"), ("st", ".h5ad"))):
            raise ValueError("Hard links require the same filesystem. Use --mode copy instead.")

    print(f"Input: {source}")
    print(f"Output: {output}")
    print(f"Mode: {mode}; slides: {len(samples)}; files: {2 * len(samples)}")
    for name, count in groups.items():
        print(f"  {name}: {count} slides")
    manifest = {
        **settings, "source_root": str(source), "output_root": str(output),
        "mode": mode, "samples": samples,
    }
    if dry_run:
        print("Dry run: no dataset directories or files created.")
        return manifest

    for name in groups:
        for folder in ("patches", "st"):
            (output / name / folder).mkdir(parents=True, exist_ok=True)
    for number, sample in enumerate(samples, 1):
        for folder, suffix in (("patches", ".h5"), ("st", ".h5ad")):
            src = source / folder / f"{sample['id']}{suffix}"
            dst = output / sample["subset"] / folder / src.name
            try:
                if mode == "hardlink":
                    os.link(src, dst)
                else:
                    shutil.copy2(src, dst)
            except OSError as exc:
                raise OSError(
                    f"Could not {mode} {src} to {dst}. Output may be incomplete. "
                    "Use a new empty output directory; use --mode copy if hard links are unsupported."
                ) from exc
        if number % 25 == 0 or number == len(samples):
            print(f"Prepared {number}/{len(samples)} slides", flush=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {output / 'manifest.json'}")
    return manifest
