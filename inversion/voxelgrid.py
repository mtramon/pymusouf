#!/usr/bin/python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
import time
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from utils.tools import print_file_datetime, check_array_order

def print_vtk_arrays(grid):
    print("\n--- PointData ---")
    pd = grid.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        print(i, pd.GetArrayName(i))
    print("\n--- CellData ---")
    cd = grid.GetCellData()
    for i in range(cd.GetNumberOfArrays()):
        print(i, cd.GetArrayName(i))
    print("\n--- FieldData ---")
    fd = grid.GetFieldData()
    for i in range(fd.GetNumberOfArrays()):
        print(i, fd.GetArrayName(i))

@dataclass
class GridGeometry:
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    mask_voxel: np.ndarray
    voxel_volume: np.ndarray
    voxel_volume_vtk: object
    density: np.ndarray | None = None

def extract_edges_fast(grid):
    dims = [0,0,0]
    grid.GetDimensions(dims)
    nx, ny, nz = dims
    pts = vtk_to_numpy(grid.GetPoints().GetData())
    pts = pts.reshape((nx, ny, nz, 3), order="F")
    x_edges = pts[:, 0, 0, 0]
    y_edges = pts[0, :, 0, 1]
    z_edges = pts[0, 0, :, 2]
    return x_edges, y_edges, z_edges

def load_voxel_grid(vtkfile):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(vtkfile))
    reader.Update()
    grid = reader.GetOutput()
    cell_data = grid.GetCellData()
    voxel_volume_vtk = cell_data.GetArray("voxel_volume")
    voxel_volume = np.array(voxel_volume_vtk)
    mask_voxel = (voxel_volume > 0).astype(np.uint8)
    # optional density field
    density = None
    density_vtk = cell_data.GetArray("density")
    if density_vtk is not None:
        density = np.array(density_vtk)

    # pts = grid.GetPoints()
    # pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    # x_edges = np.unique(pts[:, 0])
    # y_edges = np.unique(pts[:, 1])
    # z_edges = np.unique(pts[:, 2])

    x_edges, y_edges, z_edges= extract_edges_fast(grid)

    geom = GridGeometry(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        mask_voxel=mask_voxel,
        voxel_volume=voxel_volume,
        voxel_volume_vtk=voxel_volume_vtk,
        density=density,
    )
    return grid, geom



def load_dem_grid(vtkfile):
    # t0 = time.time()
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(vtkfile))
    reader.Update()

    grid = reader.GetOutput()
    # print_vtk_arrays(grid)
    # --- Read elevation from PointData ---
    elev_vtk = grid.GetPointData().GetArray("elevation")

    if elev_vtk is None:
        raise ValueError(f"No 'elevation' array found in {vtkfile}")

    # elevation = vtk_to_numpy(elev_vtk)
    elevation = np.array(elev_vtk)

    # print(f"Conversion to npy -- {time.time()-t0:.2f}")
    # --- Grid geometry ---
    extent = grid.GetExtent()
    xmin, xmax, ymin, ymax, zmin, zmax = extent
    nx, ny, nz = xmax-xmin+1, ymax-ymin+1, zmax-zmin+1
    
    # --- Convert point mask → voxel mask ---
    # reshape point array
    elevation_3d = elevation.reshape((nx, ny, nz), order="F")
    # print(f"Reshape -- {time.time()-t0:.2f}")

    # voxel mask = all 8 corners positive
    # mask_voxel_3d = (
    #     (elevation_3d[:-1, :-1, :-1] > 0) &
    #     (elevation_3d[1:,  :-1, :-1] > 0) &
    #     (elevation_3d[:-1, 1:,  :-1] > 0) &
    #     (elevation_3d[1:,  1:,  :-1] > 0) &
    #     (elevation_3d[:-1, :-1, 1:] > 0) &
    #     (elevation_3d[1:,  :-1, 1:] > 0) &
    #     (elevation_3d[:-1, 1:,  1:] > 0) &
    #     (elevation_3d[1:,  1:,  1:] > 0)
    # )
    # print(f"Mask -- {time.time()-t0:.2f}")
    # mask_voxel = mask_voxel_3d.ravel(order="F").astype(np.uint8)
    mask_voxel = (elevation > 0).astype(np.uint8)
    print(check_array_order(mask_voxel))
    # --- Extract edges ---
    # pts = grid.GetPoints()
    # pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    # print(f"GetPoint -- {time.time()-t0:.2f}")

    x_edges, y_edges, z_edges= extract_edges_fast(grid)

    # print(f"Extract edges -- {time.time()-t0:.2f}")

    geom = GridGeometry(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        mask_voxel=mask_voxel,
        voxel_volume=None,
        voxel_volume_vtk=None,
        density=None,
    )
    return grid, geom


@dataclass
class Coordinates:
    XYZ: np.ndarray
    centre: np.ndarray

def get_coordinates(geom:GridGeometry, order="F"):
    x_edges, y_edges, z_edges = geom.x_edges, geom.y_edges, geom.z_edges
    x =(x_edges[:-1] + x_edges[1:])/2
    y =(y_edges[:-1] + y_edges[1:])/2
    z =(z_edges[:-1] + z_edges[1:])/2
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    XYZ = np.vstack((X.ravel(order), Y.ravel(order), Z.ravel(order))).T
    cx =(x_edges[0] + x_edges[-1])/2
    cy =(y_edges[0] + y_edges[-1])/2
    cz =(z_edges[0] + z_edges[-1])/2
    coordinates = Coordinates(
        XYZ, 
        np.array([cx, cy, cz])
    )
    return coordinates




if __name__ =="__main__":
    from config import STRUCT_DIR
    from survey import CURRENT_SURVEY
    import sys
    survey_name = CURRENT_SURVEY.name   
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"
    dir_tel = dir_survey / "telescope"
    dir_out = dir_model / "test"
    dir_out.mkdir(parents=True, exist_ok=True)
    vs = int(sys.argv[1]) if len(sys.argv) >1 else 32  # voxel size in m (edge length)
    input_vox = dir_voxel / f"topo_voi_vox{vs}m.vts"

    # input_vox = dir_voxel / f"topo_center_anom_voi_vox{vs}m.vts"
    # grid, geom = load_voxel_grid(input_vox)

    input_dem = dir_dem / f"topo_voi.vts"
    print_file_datetime(input_dem)
    grid, geom = load_dem_grid(input_dem)


     