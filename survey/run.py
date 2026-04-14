# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from enum import Enum, auto

class RunType(Enum):
    calib = auto()
    tomo = auto()

@dataclass
class Run:
    name : str
    path : str = field(default_factory=lambda : str)
    def __str__(self): return f"Run: {self.name}"
    def __post_init__(self):
        self.path = Path(self.path)
        self.make_dirs()
        
    def make_dirs(self):
        self.dirs={}
        list_subdirs = ["raw", "reco", "log", "png", "npy", "pkl", "json"]
        for sd in list_subdirs: 
            subdir = self.path / sd
            subdir.mkdir(parents=True, exist_ok=True)
            self.dirs[sd] = subdir

class RunCalib(Run):
    def __init__(self, **kwargs):
        Run.__init__(self, **kwargs)

class RunTomo(Run):
    def __init__(self, **kwargs):    
        Run.__init__(self,  **kwargs)
   