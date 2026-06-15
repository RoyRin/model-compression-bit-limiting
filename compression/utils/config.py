"""Configuration utilities for compression."""

from pathlib import Path
from typing import Dict, Optional
import yaml


def load_compression_config(config_path: Optional[Path] = None) -> Dict:
    """Load compression configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        Dictionary with compression and optimization parameters.
    """
    # Default configuration
    default_config = {
        "compression": {
            "bit_precision": 58,
            "bits_for_encoding_count": 6,
            "min_prob": 1e-8,
            "temperature": 1.0,
        }
    }

    # Load from file if provided
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        # Merge with defaults (loaded config overrides defaults)
        if loaded_config and "compression" in loaded_config:
            default_config["compression"].update(loaded_config["compression"])

    return default_config
