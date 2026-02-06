"""Load config from config.yaml. Override path with CONFIG_PATH env var.
Loads .env from the project root (or cwd) so OPENROUTER_API_KEY can be set there."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv optional; env vars can be set in shell instead

try:
    import yaml
except ImportError:
    yaml = None

_CONFIG = None
_DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"

# Project root: directory containing config_loader.py (and config.yaml, scripts, default files)
PROJECT_ROOT = Path(__file__).resolve().parent

# Default filenames in project root (all intermediate and input/output files live here)
DECK_EXPORT = "anki.txt"
EMBEDDINGS_CSV = "anki_embeddings.csv"
LEARNING_OBJECTIVES_CSV = "learning_objectives.csv"
CARDS_CSV = "anki_cards.csv"
PROGRESS_CSV = "anki_progress.csv"
DECK_APKG = "deck.apkg"


def path_in_project(filename: str) -> Path:
    """Return path to a default file in the project root."""
    return PROJECT_ROOT / filename


def require_file(filename: str, for_script: str) -> Path:
    """Return path to file in project root; raise FileNotFoundError if missing."""
    p = path_in_project(filename)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file for {for_script}: {p}\n"
            f"Create or place {filename} in the project root: {PROJECT_ROOT}"
        )
    return p


def load_config(path=None):
    """Load config from YAML file. Returns nested dict. Cached after first load."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if yaml is None:
        raise ImportError("PyYAML is required for config. Install with: pip install pyyaml")
    config_path = path or os.environ.get("CONFIG_PATH") or _DEFAULT_PATH
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_api_key(cfg=None, env_key="OPENROUTER_API_KEY"):
    """Return API key from config (if set there) or from environment."""
    cfg = cfg or load_config()
    # Optional: support api_key in config (not recommended to commit secrets)
    key = (cfg.get("api_key") or os.environ.get(env_key) or "").strip()
    return key if key else None
