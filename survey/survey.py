# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from argparse import ArgumentTypeError
from pathlib import Path
from typing import Union

import numpy as np
import yaml

#package module(s)
from config import DATA_DIR, DEFAULT_SURVEY, SAMPLE_DIR, STRUCT_DIR, USE_SAMPLE_DATA
from telescope import DICT_TEL
from .run import Run, RunTomo, RunCalib

FYAML = Path(__file__).parent / "survey.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_survey_yaml() -> dict:
    with open(FYAML, "r") as f:
        try:
            return yaml.load(f, Loader=yaml.SafeLoader) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Failed to load survey catalog from {FYAML}") from exc


survey_yaml = _load_survey_yaml()
LIST_AVAIL_SURVEY = list(survey_yaml.keys())
DICT_SURVEY = {}


class Survey:

    def __init__(self, name: str):
        self.name = name
        self.dem = Union[Path, str]
        self.telescopes = {}
        self.runs = {}

    def __str__(self):
        sout = f"\nSurvey: {self.name}\n\n - " + f"\n - ".join(v.__str__() for _, v in self.runs.items())
        return sout

    def set_surface_grid(self):
        s = self.dem
        if isinstance(s, str):
            s = Path(s)
        if s.suffix == ".npy":
            grid = np.load(s)
        elif s.suffix == ".txt":
            grid = np.loadtxt(s)
        else:
            raise ValueError("Wrong DEM file format")
        shp = grid.shape
        if shp[0] == 3:
            grid = grid.T
        mx, my = shp[0] // 2, shp[1] // 2
        center_xy = np.array([grid[mx, my, 0], grid[my, mx, 1]])
        self.surface_center = center_xy
        self.surface_grid = grid


def _resolve_under_roots(path_value: Union[str, Path], roots: list[Path], strip_first_part: bool = False) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidate_paths = [path]
    if strip_first_part and len(path.parts) > 1:
        candidate_paths.append(Path(*path.parts[1:]))

    candidates = []
    for root in roots:
        for rel_path in candidate_paths:
            candidates.append((root / rel_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_dem_path(path_value: Union[str, Path]) -> Path:
    roots = [STRUCT_DIR, PROJECT_ROOT]
    if USE_SAMPLE_DATA:
        roots = [SAMPLE_DIR, STRUCT_DIR, PROJECT_ROOT]
    return _resolve_under_roots(path_value, roots, strip_first_part=False)


def resolve_run_path(path_value: Union[str, Path]) -> Path:
    if USE_SAMPLE_DATA:
        roots = [SAMPLE_DIR, DATA_DIR, PROJECT_ROOT]
    else:
        roots = [DATA_DIR, PROJECT_ROOT]
    return _resolve_under_roots(path_value, roots, strip_first_part=True)


def get_runs(content: dict):
    runs = {}
    for k, v in content.items():
        run_path = resolve_run_path(v)
        if "tomo" in k:
            r = RunTomo(name=k, path=run_path)
        elif "cal" in k:
            r = RunCalib(name=k, path=run_path)
        else:
            r = Run(name=k, path=run_path)
        runs[k] = r
    return runs


def set_survey(name: str):
    if name in DICT_SURVEY:
        return DICT_SURVEY[name]

    survey = Survey(name)
    if name not in LIST_AVAIL_SURVEY:
        print(f"{name} survey not in 'survey.yaml'")
        return None

    survey.dem = resolve_dem_path(survey_yaml[name]["dem"])
    DICT_SURVEY[name] = survey

    ltel = survey_yaml[name].get("telescope", {})
    for k, v in ltel.items():
        if k in DICT_TEL:
            survey.telescopes[k] = DICT_TEL[k]
        else:
            print(f"Tel {k} not in DICT_TEL")
        runs = get_runs(v.get("run", {}))
        survey.runs[k] = runs
    return survey


def str2survey(v):
    if isinstance(v, Survey):
        return v
    if v in list(DICT_SURVEY.keys()):
        return DICT_SURVEY[v]
    elif v in [f"sur_{k}" for k in list(DICT_SURVEY.keys())]:
        return DICT_SURVEY[v[4:]]
    elif v in [k.lower() for k in list(DICT_SURVEY.keys())]:
        return DICT_SURVEY[v.upper()]
    elif v in [f"survey_{k.lower()}" for k in list(DICT_SURVEY.keys())]:
        return DICT_SURVEY[v[7:].upper()]
    else:
        raise ArgumentTypeError('Input survey does not exist.')


current_survey_name = DEFAULT_SURVEY
CURRENT_SURVEY = set_survey(current_survey_name)

_DATA_SOURCE_LABEL = "sample" if USE_SAMPLE_DATA else "data"
print(f"[survey] source={_DATA_SOURCE_LABEL} | survey={CURRENT_SURVEY.name} | data_root={DATA_DIR}")

if __name__ == "__main__":
    pass