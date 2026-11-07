import argparse
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hf import load_hf_token
from utils.paths import ROOT

DEFAULT_SAMPLE_IDS = ("NCBI339","NCBI342","NCBI344","NCBI351")


def download_hest(patterns, local_dir):
    repo_id = 'MahmoodLab/hest'
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=local_dir,
        token=load_hf_token(),
    )


def main(ids_to_query=DEFAULT_SAMPLE_IDS):
    local_dir = ROOT / 'hest_data'

    list_patterns = []
    for sample_id in ids_to_query:
        list_patterns.append(f"patches/{sample_id}.h5")
        list_patterns.append(f"st/{sample_id}.h5ad")

    download_hest(list_patterns, local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HEST samples by sample ID.")
    parser.add_argument("ids", nargs="*", default=DEFAULT_SAMPLE_IDS,
                        help="Sample IDs, e.g. NCBI338 TENX95 (default: NCBI338)")
    args = parser.parse_args()
    main(args.ids)
