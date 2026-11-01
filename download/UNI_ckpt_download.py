import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def load_hf_token():
    config_path = ROOT / "config" / "token.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Create {config_path} and set hf_token.")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    token = config.get("hf_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Set hf_token in {config_path} before downloading.")
    return token.strip()


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
