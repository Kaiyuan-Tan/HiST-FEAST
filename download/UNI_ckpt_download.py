from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hf import load_hf_token
from utils.paths import ROOT


def main():
    checkpoint_path = hf_hub_download(
        repo_id="MahmoodLab/UNI",
        filename="pytorch_model.bin",
        local_dir=ROOT / "ckpts",
        token=load_hf_token(),
        force_download=True,
    )
    print(f"Saved UNI checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
