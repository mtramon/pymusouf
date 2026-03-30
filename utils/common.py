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
        self.dirs = {}
        self.setup()
        
    def setup(self):
        """
        Setup survey object, analyse and final file paths 
        """
        if isinstance(self.args.survey, str) : self.survey = str2survey(self.args.survey)
        if isinstance(self.args.telescope, str) : self.telescope = str2telescope(self.args.telescope)
        if "run" in self.args : self.run = self.survey.runs[self.telescope.name][self.args.run]


