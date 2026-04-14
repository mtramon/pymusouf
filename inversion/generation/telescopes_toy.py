#!/usr/bin/python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import inset_locator
import vtk
from tqdm import tqdm
import pickle
import scipy.sparse as sp
import sys

# package module(s)
from config import STRUCT_DIR
from telescope import tel_SNJ
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
from utils.vtk.convert_telpos_to_vtk import adjust_detectors_height, save_detectors_vtk

def sample_points_annulus(ns, rmin, rmax, center=(0,0), dmin=10, max_trials=100000):
    cx, cy = center
    points = []
    thetas = []
    radii = []
    trials = 0
    while len(points) < ns and trials < max_trials:
        trials += 1
        theta = rng.uniform(0, 2*np.pi)
        # r = rng.uniform(rmin, rmax) #bias towards inner edge
        r = np.sqrt(rng.uniform(rmin**2, rmax**2))
        x = cx + r*np.cos(theta)
        y = cy + r*np.sin(theta)
        if not points:
            points.append((x,y))
            thetas.append(theta)
            radii.append(r)
            continue
        pts = np.array(points)
        dist = np.sqrt((pts[:,0]-x)**2 + (pts[:,1]-y)**2)
        if np.all(dist >= dmin):
            points.append((x,y))
            thetas.append(theta)
            radii.append(r)
    return np.array(points), np.array(thetas), np.array(radii)


if __name__ == "__main__" : 
    survey_name = CURRENT_SURVEY.name   
    print(f"Processing survey: {survey_name}")
    print(f"Survey structure directory: {STRUCT_DIR / survey_name}")
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_png = dir_survey / "png"
    dir_tel = dir_survey / "telescope"
    dir_png.mkdir(parents=True, exist_ok=True)
    vs = int(sys.argv[1]) if len(sys.argv) >1 else 8  # voxel size in m (edge length)
    input_file = dir_voxel / f"topo_voi_vox{vs}m.vts"
    # input_file = dir_voxel / f"rect_grid_vox{vs}m.vts"
    # input_file = dir_voxel / f"ElecCond_CentralCube_aligned_voi_vox{vs}m.vts"

    print_file_datetime(input_file)

    # Read source structured grid
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(input_file))
    reader.Update()
    grid = reader.GetOutput()
    voxel_volume_vtk = reader.GetOutput().GetCellData().GetArray("voxel_volume")
    voxel_volume = np.array(voxel_volume_vtk)
    mask_voxel = voxel_volume > 0.0   # True = voxel du volcan
    mask_voxel = mask_voxel.astype(np.uint8)  # Numba-friendly
    # assert V.shape[0] == nx * ny * nz, "Voxel volume array size does not match grid dimensions"
    print(f"Voxel volume array shape: {voxel_volume.shape}")
    xmin, xmax, ymin, ymax, zmin, zmax = grid.GetBounds()
    pts = grid.GetPoints()
    pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    x_edges = np.unique(pts[:, 0])
    y_edges = np.unique(pts[:, 1])
    z_edges = np.unique(pts[:, 2])
    x_center, y_center = (xmin+xmax) / 2, (ymin+ymax) / 2
    print(f"Grid dimensions: {len(x_edges)-1} x {len(y_edges)-1} x {len(z_edges)-1} voxels")

    seed = 9506
    rng = np.random.default_rng(seed)
    ntoy = 15
    radius_range = [450, 500]
    xy_samples, thetas_samples, radii_samples = sample_points_annulus(ntoy, radius_range[0], radius_range[1], (x_center, y_center), dmin=150, max_trials=1e6)
    tel0 = tel_SNJ
    z0 = np.ones(ntoy)*tel0.coordinates[2]
    print(f"xy_samples : {xy_samples}, {xy_samples.shape}")
    positions_sample = np.vstack((xy_samples[:,0], xy_samples[:,1], z0)).T
    print(positions_sample.shape)
    new_pos, moved = adjust_detectors_height(
        positions_sample,
        x_edges, y_edges, z_edges,
        mask_voxel,
        offset=1.0
    )
    fout_vtk = dir_tel/ f"toy_telescopes_s{seed}_vox{vs}m.vtp"
    save_detectors_vtk(
        positions_sample,#[0:2],#[np.newaxis],
        new_pos,#[0:2],#[np.newaxis],
        moved,
        fout_vtk
    )
    print(f"Saved {fout_vtk}")
    # print(thetas_samples)
    dict_toy = {}
    ix_sort = np.argsort(thetas_samples)
    thetas_sort = thetas_samples[ix_sort]
    new_pos_sort = new_pos[ix_sort]
    for i, xyz in enumerate(new_pos_sort):
        # if (i!=0) & (i != 7): continue 
        name = f"toy_{i}"
        toy = deepcopy(tel0)
        toy.name = name
        toy.coordinates = xyz
        toy.azimuth = (3*np.pi/2-thetas_sort[i])*180/np.pi 
        # print(i, thetas_samples[i]*180/np.pi , toy.azimuth)
        # toy.compute_angular_coordinates()
        dict_toy[name] = toy
    fout_pkl = dir_tel / f"{fout_vtk.stem}.pkl"
    # dict_toy = CURRENT_SURVEY.telescopes
    # fout_pkl = dir_tel / f"true_telescopes_vox{vs}m.pkl"
    with open(fout_pkl, 'wb') as f:
        pickle.dump(dict_toy, f)
    print(f"Saved {fout_pkl}")
    
    ###TEST
    # with open(fout_pkl, 'rb') as f:
    #     dict_toy = pickle.load(f) 
    # toy0 = dict_toy["toy_0"]
    # print(toy0.azimuth)
    # toy0.compute_angular_coordinates()
    # print(toy0.azimuth_matrix["3p1"])
    ###