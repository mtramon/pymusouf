#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np 
from pathlib import Path
from utils.tools import print_file_datetime
import vtk
from tqdm import tqdm
#package module(s)    
dir_dem = Path("/Users/raphael/pymusouf/struct_link/soufriere/dem")
# file = dir_dem / "soufriere_dome_surface_5m.npy"
file = dir_dem / "topo.npy"

# file = dir_dem / "soufriere_voi_surface_5m.npy"

surface = np.load(file) if str(file) == str(dir_dem/"topo.npy") else np.load(file).T
print_file_datetime(file)
basename = "_".join(file.stem.split("_")[:2])
nx, ny, _ = surface.shape
print(nx, ny)
x = surface[:, :, 0]
y = surface[:, :, 1]
z_surface = surface[:, :, 2]
# Niveau de base
z_base = z_surface.min() - 5
nx, ny = z_surface.shape
# Résolution verticale (à ajuster)
print(np.diff(x[:,0].ravel()))
res = np.median(np.diff(x[:,0].ravel())) #m 
print("res = ", res)
# res = 1
print("z_max = ", z_surface.max())
nz = int((z_surface.max() - z_base) // res)
print("nz = ", nz)
z_levels = np.linspace(z_base, z_surface.max(), nz)


# Créer la grille structurée
grid = vtk.vtkStructuredGrid()
grid.SetDimensions(nx, ny, 1)

# Créer les points
points = vtk.vtkPoints()
points.SetNumberOfPoints(nx * ny * nz)
def idx(i, j, k):
    return i + nx * (j + ny * k)
pid = 0
for k in tqdm(range(nz), total=nz, desc="Point"):
    for j in range(ny):
        for i in range(nx):
            points.SetPoint(
                pid,
                x[i,j],
                y[i, j],
                z_levels[k]
            )
            pid += 1

grid = vtk.vtkStructuredGrid()
grid.SetDimensions(nx, ny, nz)
grid.SetPoints(points)
solid = vtk.vtkUnsignedCharArray()
solid.SetName("solid")
solid.SetNumberOfTuples(nx * ny * nz)
elevation = vtk.vtkFloatArray()
elevation.SetName("elevation")
elevation.SetNumberOfValues(nx * ny *nz)
pid = 0
for k in tqdm(range(nz), total=nz, desc="Value"):
    for j in range(ny):
        for i in range(nx):
            solid.SetValue(
                pid, 1 if z_levels[k] <= z_surface[i, j] else 0
            )
            elevation.SetValue(pid, surface[i, j, 2])
            pid += 1

grid.GetPointData().AddArray(solid)
grid.GetPointData().AddArray(elevation)


threshold = vtk.vtkThreshold()
threshold.SetInputData(grid)
threshold.SetInputArrayToProcess(
    0, 0, 0,
    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
    "solid"
)
threshold.SetLowerThreshold(1)
threshold.SetUpperThreshold(1)
threshold.Update()
ugrid = threshold.GetOutput()
print(ugrid.GetNumberOfCells())
print(ugrid.GetNumberOfPoints())
writer = vtk.vtkXMLUnstructuredGridWriter()
file_out = dir_dem / f"{basename}_volume_{int(res)}m.vtu"
writer.SetFileName(file_out)
writer.SetInputData(ugrid)
writer.Write()
print(f"Save {file_out}")
