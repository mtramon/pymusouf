#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import gzip
import glob
import os
from pathlib import Path



class RawData:

    def __init__(self, path:Path):
        self.path   = path  
        self.dataset: list[Path] = []
        self.nfiles_tot = 0 
    def __str__(self):
        return f"RawData {self.path} - nfiles : {len(self.dataset)}"

    def is_gz_file(self, file: str | Path) -> bool:
        return str(file).endswith(".gz")

    def listdatafiles(self, path: Path, nfiles=None, fmt="dat"):
        files = sorted(path.glob(f"*{fmt}*"))
        self.nfiles_tot = len(files)
        if nfiles is None or int(nfiles) <= 0:
            return files
        return files[:nfiles]

    def fill_dataset(self, **args):
        if self.path.is_file():
            self.dataset = [Path(self.path)]    
        elif self.path.is_dir(): 
            self.dataset = self.listdatafiles(self.path, **args)
        else: raise Exception("Wrong 'path' object.")
    
    def readfile(self, file: str | Path):
        file = Path(file)
        opener = gzip.open if file.suffix == ".gz" else open

        try:
            with opener(file, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    yield line
        except OSError as e:
            raise ValueError(f"Invalid data file: {file}") from e
        

if __name__ == "__main__":

    # path = Path("../rawdata_OM_calib")
    # raw_data = RawData(path=path)
    # raw_data.fill_dataset()


    path = Path("../../data/BR/Calib10/raw")
    raw_data = RawData(path=path)
    raw_data.fill_dataset()

    # nlines=0
    # for file in raw_data.dataset:
    #     lines = raw_data.readfile(file)
    #     nlines += len(lines)

    # print(f"nlines = {nlines}")


"""
      
    def listFilesUrl(self, url:str):
        '''
        List files at 'url' to be fetched
        '''
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        listUrl = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('')]
        return listUrl


 
    def fetchFiles(self, nfiles:int, save_path:str):
        '''
        Fetch datafiles online at 'url'
        '''
        Path(save_path).mkdir(parents=True, exist_ok=True)   
         #with extension '.dat.gz'
        for i, url in enumerate(self.listFilesUrl()):
            if i > nfiles: break 
            file_basename = os.path.basename(url)
            if not os.path.isfile(os.path.join(save_path, "", file_basename)):
            #check if already existing file 
                file = requests.get(url)
                with open(os.path.join(save_path, "" ,file_basename), 'wb') as f:
                    f.write(file.content)
                f.close()
        else : pass 
   


"""