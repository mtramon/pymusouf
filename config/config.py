# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from copy import deepcopy
import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent
PACKAGE_ROOT = CONFIG_DIR.parent


def _find_runtime_root() -> Path:
    env_root = os.getenv("PYMUSOUF_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (
            (base / "config" / "config.yaml").exists()
            or (base / "data_link").exists()
            or (base / "struct_link").exists()
            or (base / "sample").exists()
        ):
            return base

    return PACKAGE_ROOT


PROJECT_ROOT = _find_runtime_root()

DEFAULT_CONFIG = {
    "default_survey": "soufriere",
    "paths": {
        "data": "./data_link",
        "structure": "./struct_link",
        "sample": "./sample",
    },
    "runtime": {
        "use_sample_data": True,
        "create_missing_dirs": True,
    },
}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        try:
            return yaml.load(f, Loader=yaml.SafeLoader) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Invalid YAML configuration in {path}") from exc


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_legacy_keys(config: dict) -> dict:
    config = dict(config)
    config.setdefault("paths", {})

    if "data_dir" in config:
        config["paths"]["data"] = config["data_dir"]
    if "struct_dir" in config:
        config["paths"]["structure"] = config["struct_dir"]
    if "sample_dir" in config:
        config["paths"]["sample"] = config["sample_dir"]

    return config


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)

    for filename in ["config.yaml", "config.local.yaml"]:
        path = CONFIG_DIR / filename
        file_config = _normalize_legacy_keys(_read_yaml(path))
        _deep_update(config, file_config)

    env_data = os.getenv("PYMUSOUF_DATA_DIR")
    env_struct = os.getenv("PYMUSOUF_STRUCT_DIR")
    env_sample = os.getenv("PYMUSOUF_SAMPLE_DIR")

    if env_data:
        config["paths"]["data"] = env_data
    if env_struct:
        config["paths"]["structure"] = env_struct
    if env_sample:
        config["paths"]["sample"] = env_sample

    return config


CONFIG = load_config()
DEFAULT_SURVEY = CONFIG.get("default_survey", "soufriere")

DATA_DIR = _resolve_path(CONFIG["paths"]["data"])
STRUCT_DIR = _resolve_path(CONFIG["paths"]["structure"])
SAMPLE_DIR = _resolve_path(CONFIG["paths"].get("sample", "./sample"))

USE_SAMPLE_DATA = bool(CONFIG.get("runtime", {}).get("use_sample_data", True))
CREATE_MISSING_DIRS = bool(CONFIG.get("runtime", {}).get("create_missing_dirs", True))

if CREATE_MISSING_DIRS:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def use_paths():
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEFAULT_SURVEY: {DEFAULT_SURVEY}")
    print(f"DATA_DIR: {DATA_DIR}, exists: {DATA_DIR.exists()}")
    print(f"STRUCT_DIR: {STRUCT_DIR}, exists: {STRUCT_DIR.exists()}")
    print(f"SAMPLE_DIR: {SAMPLE_DIR}, exists: {SAMPLE_DIR.exists()}")
    print(f"USE_SAMPLE_DATA: {USE_SAMPLE_DATA}")


if __name__ == "__main__":
    use_paths()

