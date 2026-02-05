#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np 
from pathlib import Path
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
#package module(s)    
from utils.tools import print_file_datetime

dir_survey = Path("/Users/raphael/structure/soufriere")
dir_dem = dir_survey/"dem"
dir_voxel =dir_survey /"voxel"
# file = dir_voxel / "topo_voi_vox8m.vtu"
vs = 8  #voxel size in m
input_file = dir_voxel / f"topo_voi_vox{vs}m.vtu"
print_file_datetime(input_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(input_file)
reader.Update()
ugrid = reader.GetOutput()

cell_locator = vtk.vtkStaticCellLocator()
cell_locator.SetDataSet(ugrid)
cell_locator.BuildLocator()

ncells = ugrid.GetNumberOfCells()
bboxes = np.zeros((ncells, 6))  # xmin,xmax,ymin,ymax,zmin,zmax

for i in tqdm(range(ncells), desc="Cells "):
    cell = ugrid.GetCell(i)
    pts = cell.GetPoints()
    xs, ys, zs = [], [], []
    for j in range(pts.GetNumberOfPoints()):
        x,y,z = pts.GetPoint(j)
        xs.append(x); ys.append(y); zs.append(z)
    bboxes[i] = [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]

centers = 0.5 * (bboxes[:,[0,2,4]] + bboxes[:,[1,3,5]])


###Specify anomaly VOI from another VTU file
input_file = dir_dem / "soufriere_bulge_volume_5m.vtu"
print_file_datetime(input_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(input_file)
reader.Update()
ugrid_bulge = reader.GetOutput()

# world-coordinate bounds (xmin,xmax,ymin,ymax,zmin,zmax)
xmin, xmax, ymin, ymax, zmin, zmax = ugrid_bulge.GetBounds()
print(f"Bounds: {xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}")
nx, ny, nz = xmax-xmin+1, ymax-ymin+1, zmax-zmin+1

mask = (
    (centers[:,0] >= xmin) & (centers[:,0] <= xmax) &
            (centers[:,1] >= ymin) & (centers[:,1] <= ymax)  &  
             (centers[:,2] >= zmin) & (centers[:,2] <= zmax)
)
all_cell_ids = np.arange(ncells)
bulge_cell_ids = np.where(mask)[0]
print(f"Number of voxels in bulge VOI: {len(bulge_cell_ids)}")
dome_cell_ids = np.where(~mask)[0]
print(f"Number of voxels in dome VOI: {len(dome_cell_ids)}")

extract = vtk.vtkExtractCells()
extract.SetInputData(ugrid)
id_list = vtk.vtkIdList()
dens_arr = vtk.vtkDoubleArray()
dens_arr.SetName("density")
# density = []
dens_bulge=1.0
dens_dome=2.0
# build density array in numpy and convert once to vtk (much faster than per-cell InsertNextValue)
density = np.where(mask, dens_bulge, dens_dome).astype(np.float64)
dens_arr = numpy_support.numpy_to_vtk(density, deep=True, array_type=vtk.VTK_DOUBLE)
dens_arr.SetName("density")
# populate id_list efficiently
id_list.SetNumberOfIds(ncells)
for i in range(ncells):
    id_list.SetId(i, int(i))
extract.SetCellList(id_list)
extract.Update()
subgrid = extract.GetOutput()
writer = vtk.vtkXMLUnstructuredGridWriter()
subgrid.GetCellData().AddArray(dens_arr)
output_vtu =  dir_dem.parent/ "voxel" / f"topo_bulge_voi_vox{int(vs)}m.vtu"
writer.SetFileName(output_vtu)
writer.SetInputData(subgrid)
writer.Write()
print(f"Save {output_vtu}")


