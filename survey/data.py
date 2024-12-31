#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum, auto
import os
from pathlib import Path
import inspect
import logging
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
import gzip
import glob
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class DataType(Enum):
    real = auto()
    mc = auto()


class DataSet:
    @abstractmethod
    def dataset(self, dataset):
        pass

    

@dataclass
class BaseData : 
   
    path : Path


class RawData(BaseData):


    def __init__(self, path:Path):

        BaseData.__init__(self, path)      
        self.dataset = DataSet


    def __str__(self):

        return f"RawData {self.path} - nfiles : {len(self.dataset)}"


    def is_gz_file(self, filepath):

        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'


    def listdatafiles(self, path, fmt:str="dat", max_nfiles:int=None): 
            
        nfiles = len(os.listdir(path))
        if max_nfiles is not None: nfiles = max_nfiles 
        dataset = [f for n, f in enumerate(glob.glob(str(path) + f'/*{fmt}*')) if n < nfiles ]
       
        return dataset


    def fill_dataset(self, **args):

        if self.path.is_file():
            self.dataset = [self.path]
                   
        elif self.path.is_dir(): 
            self.dataset = self.listdatafiles(self.path, **args)
            
        else: raise Exception("Wrong 'path' object.")

    
    def readfile(self, file):
        
        # logging.debug("survey/data.py: Reading file %s", file)

        datalines = list()
        
        try:
            if str(file).endswith('.npy'):
                data = np.load(file, allow_pickle=True)
                if isinstance(data, np.ndarray):
                    datalines.extend(data.tolist())
                else:
                    datalines.append(data.tolist())
                
            elif self.is_gz_file(file):
                with gzip.open(f"{file}", 'rt') as f:
                    for l in f:
                        if l == "\n": continue
                        datalines.append(l) 
                        
            else:
                with open(f"{file}", 'rt') as f:
                    for l in f:
                        if l == "\n": continue
                        datalines.append(l) 

        except OSError: 
            raise ValueError("Data files should be either .txt, .dat, .npy format, or in compressed gunzip form '.gz' ")

        # logging.debug(f"survey/data.py: Read {len(datalines)} lines from {file}")

        return datalines
    
    

if __name__ == "__main__":

    path = Path("../rawdata_OM_calib")
    raw_data = RawData(path=path)
    raw_data.fill_dataset()


    path = Path("../data/BR/Calib10/rawdata")
    raw_data = RawData(path=path)
    raw_data.fill_dataset()

    # nlines=0
    # for file in raw_data.dataset:
    #     lines = raw_data.readfile(file)
    #     nlines += len(lines)

    # print(f"nlines = {nlines}")
