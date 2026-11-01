import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def load_hf_token():
    config_path = ROOT / "config" / "token.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Create {config_path} using config/config.example.json."
        )
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    token = config.get("hf_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Set hf_token in {config_path} before downloading.")
    return token.strip()


def download_hest(patterns, local_dir):
    repo_id = 'MahmoodLab/hest'
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=local_dir,
        token=load_hf_token(),
    )


def main():
    local_dir = ROOT / 'hest_data'
    ids_to_query = ['TENX95', 'TENX96', 'TENX97', 'TENX98', 'TENX99']

    list_patterns = []
    for sample_id in ids_to_query:
        list_patterns.append(f"patches/{sample_id}.h5")
        list_patterns.append(f"st/{sample_id}.h5ad")

    download_hest(list_patterns, local_dir)


if __name__ == "__main__":
    main()
