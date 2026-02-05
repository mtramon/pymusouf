#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

#package module(s)
from telescope import str2telescope
from survey import str2survey

class Common: 
    """
    Class to set survey, telescope objects, and different paths.
    """
    def __init__(self, args:argparse.Namespace):
        self.args = args
        self.setup()
        
    def setup(self):
        """
        Setup survey object, analyse and final file paths 
        """
        if isinstance(self.args.survey, str) : self.survey = str2survey(self.args.survey)
        if isinstance(self.args.telescope, str) : self.telescope = str2telescope(self.args.telescope)
        p  = self.survey.runs[self.telescope.name][self.args.run].path
        self.data_path = p
        self.raw_path = p / "raw"
        self.reco_path = p / "reco" 
        self.reco_path.mkdir(parents=True, exist_ok=True)
        self.log_path = p / "log" 
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.plot_path = p / "plots" 
        self.plot_path.mkdir(parents=True, exist_ok=True)
        self.npy_path = p / "npy" 
        self.npy_path.mkdir(parents=True, exist_ok=True)
        self.pkl_path = p / "pkl" 
        self.pkl_path.mkdir(parents=True, exist_ok=True)



