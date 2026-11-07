"""Local Hugging Face credential configuration shared by download scripts."""
import json

from utils.paths import ROOT


def load_hf_token():
    config_path = ROOT / "config" / "token.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Create {config_path} and set hf_token.")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    token = config.get("hf_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Set hf_token in {config_path} before downloading.")
    return token.strip()
