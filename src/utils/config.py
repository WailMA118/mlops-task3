"""
Central configuration loader.

Every module in this project reads paths and parameters through `get_config()`
instead of hardcoding them. Values in config/config.yaml can be overridden by
environment variables of the form CONFIG__SECTION__SUBSECTION__KEY (double
underscore separated, case-insensitive), which is how Docker/Compose/CI inject
secrets and environment-specific values (e.g. CONFIG__DATABASE__URL).

Usage:
    from src.utils.config import get_config
    cfg = get_config()
    cfg.paths.data.train          # attribute access
    cfg["paths"]["data"]["train"] # dict-style access also works
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = two levels up from this file (src/utils/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

ENV_PREFIX = "CONFIG__"


class ConfigDict(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config_dict(v) for v in obj]
    return obj


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply CONFIG__A__B__C=value environment variables on top of the YAML config.
    Values are parsed with yaml.safe_load so "5000" -> int, "true" -> bool, etc.
    """
    config = copy.deepcopy(config)
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX):].lower().split("__")
        node = config
        for part in path[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        try:
            parsed_value = yaml.safe_load(env_value)
        except yaml.YAMLError:
            parsed_value = env_value
        node[path[-1]] = parsed_value
    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the repo root."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@lru_cache(maxsize=None)
def _load_raw_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _apply_env_overrides(raw)


def get_config(config_path: str | Path | None = None) -> ConfigDict:
    """
    Load (and cache) the project configuration.

    Parameters
    ----------
    config_path : optional override of the config file location. Defaults to
        config/config.yaml at the repo root, or the CONFIG_PATH env var if set.
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    raw = _load_raw_config(str(config_path))
    return _to_config_dict(raw)


def clear_config_cache() -> None:
    """Mainly for tests: force the next get_config() call to re-read the file."""
    _load_raw_config.cache_clear()
