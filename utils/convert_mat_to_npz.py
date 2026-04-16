

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm
#package module(s)
from survey import CURRENT_SURVEY
# from raypath import  AcqVars

if __name__ == "__main__":
    
    souf_survey = CURRENT_SURVEY
    # tel_name = "SNJ"
    # tel = souf_survey.telescope[tel_name]
    dir_path = Path("/Users/raphael/pymusouf/files/survey/soufriere")
    # dir_path = souf_survey.path
    for tel_name, tel in tqdm(souf_survey.telescope.items(), desc="Raypath"):
        print(tel_name)
        for conf_name, conf in tel.configurations.items():
            dir_acq = dir_path / "telescope" / tel_name / "acqvars"/f"az{tel.azimuth}ze{tel.zenith}"
            file_mat = dir_acq / f"acqVars_{conf_name[:2]}.mat"
            if not file_mat.exists(): 
                print(f"{file_mat} does not exist")
                continue
            acqvars = loadmat(str(file_mat))
            # Filtrer uniquement les vraies variables MATLAB
            arrays = {
                k: v for k, v in acqvars.items()
                if not k.startswith("__")
            }
            # Sauvegarder dans un fichier .npz en conservant les clés
            file_out = dir_acq / str(str(file_mat.stem)+".npz")
            np.savez(file_out, **arrays)
            print(f"Save {file_out}")