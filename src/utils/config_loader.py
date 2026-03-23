"""Load and merge YAML experiment configs."""
from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Merge with base_config if 'defaults' key present
    if "defaults" in cfg:
        base_path = path.parent / "base_config.yaml"
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base

    return cfg
