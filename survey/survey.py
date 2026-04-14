# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from argparse import ArgumentTypeError
import numpy as np
from pathlib import Path
from typing import Union
import yaml

#package module(s)
from telescope import DICT_TEL
from .run import Run, RunTomo, RunCalib

fyaml = Path(__file__).parent / "survey.yaml"
with open( fyaml ) as f: #same
    try:
        survey_yaml = yaml.load(f, Loader=yaml.SafeLoader) # The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
    except yaml.YAMLError as exc:
        print(exc)

LIST_AVAIL_SURVEY = list(survey_yaml.keys())
DICT_SURVEY = {}
    
class Survey: 
   
    def __init__(self, name:str):
        self.name = name
        self.dem = Union[Path, str] #path to dem file
        self.telescopes = {} #tel: Telescope object
        self.runs = {} #tel: path to data runsfiles
        # self.surface_grid = np.ndarray #shape : (3, m, n)
        # self.surface_center = None
        # self.flux = {}
        # self.raypath = {}

    def __str__(self): 
        sout = f"\nSurvey: {self.name}\n\n - "+ f"\n - ".join(v.__str__() for _,v in self.runs.items())
        return sout

    def set_surface_grid(self): 
        """_summary_
        """
        s = self.dem
        if isinstance(s,str): s=Path(s) 
        if s.suffix == ".npy": grid = np.load(s)  #grid (np.ndarray): surface grid shape (3, m, n)
        elif s.suffix ==".txt": grid = np.loadtxt(s)
        else: raise ValueError("Wrong DEM file format")
        shp = grid.shape
        if shp[0]==3: grid=grid.T
        mx, my = shp[0]//2, shp[1]//2
        center_xy = np.array([grid[mx, my, 0], grid[my, mx, 1]])
        self.surface_center = center_xy
        self.surface_grid = grid

def get_runs(content:dict):
    """"
    Get available data runs in 'content' dict
    """
    runs = {}
    for k, v in content.items():
        r = None
        if "tomo" in k: r= RunTomo(name=k, path=v ) 
        elif "cal" in k: r= RunCalib(name=k, path=v ) 
        else : r= Run(name=k, path=v ) 
        runs[k] = r
    return runs

def set_survey(name:str):
    survey = Survey(name)
    if not name in LIST_AVAIL_SURVEY: 
        print(f"{name} survey not in 'survey.yaml'")
        return None
    f = Path(survey_yaml[name]["dem"])
    # if f.exists():
    #     try : 
    #         survey.dem = f 
    #         survey.set_surface_grid()
    #     except : 
    #         print(f"Failed to set surface grid for {name} survey.")
    # else: 
    #     print(f"DEM file not available for {name} survey.")
    DICT_SURVEY[name] = survey
    ltel = survey_yaml[name]["telescope"]
    for k,v in ltel.items(): 
        if k in DICT_TEL: survey.telescopes[k] = DICT_TEL[k] 
        else: print(f"Tel {k} not in DICT_TEL")
        runs = get_runs(v["run"])
        survey.runs[k] = runs
        # if v["flux"] is not None : survey.flux[k] = v["flux"] 
        # if v["raypath"] is not None : survey.raypath[k] = v["raypath"] 
    return survey

def str2survey(v):
    if isinstance(v, Survey):
       return v
    if v in list(DICT_SURVEY.keys()):
        return DICT_SURVEY[v]
    elif v in [f"sur_{k}" for k in list(DICT_SURVEY.keys()) ]:
        return DICT_SURVEY[v[4:]]
    elif v in [ k.lower() for k in list(DICT_SURVEY.keys())]:
        return DICT_SURVEY[v.upper()]
    elif v in [f"survey_{k.lower()}" for k in list(DICT_SURVEY.keys()) ]:
        return DICT_SURVEY[v[4:].upper()]
    else:
        raise ArgumentTypeError('Input survey does not exist.')

current_survey_name = "soufriere"
CURRENT_SURVEY = set_survey(current_survey_name)

if __name__=="__main__":
    pass