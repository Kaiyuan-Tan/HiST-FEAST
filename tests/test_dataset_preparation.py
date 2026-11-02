import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_files import paired_samples, project_path, ROOT
from scripts.separate_species import main as separate_species, species_plan
from scripts.split_dataset import main as split_dataset, split_counts, split_plan


class DatasetPreparationTests(unittest.TestCase):
    def make_dataset(self, root, count=10):
        (root / "patches").mkdir(parents=True)
        (root / "st").mkdir()
        for i in range(count):
            (root / "patches" / f"sample{i:03}.h5").write_bytes(f"patch-{i}".encode())
            (root / "st" / f"sample{i:03}.h5ad").write_bytes(f"expression-{i}".encode())

    def make_metadata(self, path, count=10):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "species"])
            writer.writeheader()
            for i in range(count):
                writer.writerow({"id": f"sample{i:03}",
                                 "species": "Homo sapiens" if i < 3 else "Mus musculus"})

    def test_species_cli_preserves_pairs_and_original_files_with_hardlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, output, metadata = base / "source", base / "species", base / "meta.csv"
            self.make_dataset(root)
            self.make_metadata(metadata)
            with contextlib.redirect_stdout(io.StringIO()):
                separate_species(["--root", str(root), "--output-dir", str(output),
                                  "--metadata", str(metadata)])
            self.assertEqual(len(paired_samples(root)), 10)
            self.assertEqual(len(paired_samples(output / "human")), 3)
            self.assertEqual(len(paired_samples(output / "mouse")), 7)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            for sample in manifest["samples"]:
                for folder, suffix in (("patches", ".h5"), ("st", ".h5ad")):
                    src = root / folder / f"{sample['id']}{suffix}"
                    dst = output / sample["subset"] / folder / src.name
                    self.assertTrue(os.path.samefile(src, dst))
                    self.assertEqual(src.read_bytes(), dst.read_bytes())

    def test_split_cli_copies_complete_disjoint_slides_in_7_2_1_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "source", Path(temp) / "split"
            self.make_dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataset(["--root", str(root), "--output-dir", str(output), "--mode", "copy"])
            groups = {name: set(paired_samples(output / name)) for name in ("train", "val", "test")}
            self.assertEqual([len(ids) for ids in groups.values()], [7, 2, 1])
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])
            self.assertEqual(set.union(*groups.values()), set(paired_samples(root)))
            for subset, ids in groups.items():
                for sample_id in ids:
                    for folder, suffix in (("patches", ".h5"), ("st", ".h5ad")):
                        src = root / folder / f"{sample_id}{suffix}"
                        dst = output / subset / folder / src.name
                        self.assertEqual(src.read_bytes(), dst.read_bytes())
                        self.assertFalse(os.path.samefile(src, dst))

    def test_rounding_seed_and_nonempty_small_splits(self):
        self.assertEqual(split_counts(171), {"train": 120, "val": 34, "test": 17})
        self.assertEqual(split_counts(43), {"train": 30, "val": 9, "test": 4})
        for count in range(3, 100):
            counts = split_counts(count)
            self.assertEqual(sum(counts.values()), count)
            self.assertTrue(all(value >= 1 for value in counts.values()))
        with self.assertRaises(ValueError):
            split_counts(2)
        ids = [f"id{i:03}" for i in range(50)]
        self.assertEqual(split_plan(ids, 42), split_plan(list(reversed(ids)), 42))
        self.assertNotEqual(split_plan(ids, 42), split_plan(ids, 7))

    def test_missing_pairs_and_metadata_stop_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "source", Path(temp) / "output"
            self.make_dataset(root)
            (root / "st" / "sample000.h5ad").unlink()
            with self.assertRaisesRegex(ValueError, "Unpaired slides"):
                split_dataset(["--root", str(root), "--output-dir", str(output)])
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "absent from metadata"):
            species_plan(["missing"], {})

    def test_mixed_species_are_not_misclassified(self):
        metadata = {
            "human": {"species": "Homo sapiens"},
            "mouse": {"species": "Mus musculus"},
            "mixed": {"species": "Homo sapiens/Mus musculus"},
        }
        with contextlib.redirect_stdout(io.StringIO()):
            samples, counts = species_plan(list(metadata), metadata)
        self.assertEqual(counts, {"human": 1, "mouse": 1, "other": 1})
        self.assertEqual(next(s["subset"] for s in samples if s["id"] == "mixed"), "other")

    def test_dry_run_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "source", Path(temp) / "output"
            self.make_dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataset(["--root", str(root), "--output-dir", str(output), "--dry-run"])
            self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output / "existing.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                split_dataset(["--root", str(root), "--output-dir", str(output)])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_relative_paths_are_based_on_project_root(self):
        self.assertEqual(project_path("hest_data"), (ROOT / "hest_data").resolve())


if __name__ == "__main__":
    unittest.main()
