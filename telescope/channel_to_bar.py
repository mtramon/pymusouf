import json 
import numpy as np
grid = np.array([
    [1,9,1,9,1,9,1,9],
    [2,10,2,10,2,10,2,10],
    [3,11,3,11,3,11,3,11],
    [4,12,4,12,4,12,4,12],
    [5,13,5,13,5,13,5,13],
    [6,14,6,14,6,14,6,14],
    [7,15,7,15,7,15,7,15],
    [8,16,8,16,8,16,8,16],
])

nx, ny = grid.shape
result = {}

for i, row in enumerate(grid):
    for j, val in enumerate(row):
        prefix = "Y" if j in [0,1,4,5] else "X"
        idx = j + ny * i
        result[str(idx)] = f"{prefix}{val}"

fout = "mapping.json"
with open(fout, "w") as f:
    json.dump(result, f, indent=2)