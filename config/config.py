# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from pathlib import Path
import yaml

with open( Path(__file__).parent / "config.yaml") as f: #same
    try:
        config = yaml.load(f, Loader=yaml.SafeLoader) # The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
    except yaml.YAMLError as exc:
        print(exc) 

DATA_DIR = Path(config["data_dir"])
STRUCT_DIR = Path(config["struct_dir"])

def use_paths():
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"STRUCT_DIR: {STRUCT_DIR}")

if __name__ == "__main__":
    use_paths()

