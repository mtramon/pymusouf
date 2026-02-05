#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numba
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import vtk
from tqdm import tqdm
import scipy.sparse as sp

# package module(s)
from cli import get_common_args
from utils.common import Common
from utils.tools import print_file_datetime

def angular_acceptance_map(
        u,
        v,
        delta_z,
        Lx=80.0,
        Ly=80.0,
    ):
        """
        Acceptance géométrique normalisée A(u,v)
        selon Thomas & Willis (rectangles).

        u, v : centres bins (dx/dz, dy/dz) ou (tan(theta_x), tan(theta_y))
        delta_z : séparation entre plans extrêmes (cm)
        """
        # centres des bins
        U, V = np.meshgrid(u, v, indexing="ij")
        Ax = np.maximum(0.0, 1.0 - np.abs(U) * delta_z / Lx)
        Ay = np.maximum(0.0, 1.0 - np.abs(V) * delta_z / Ly)
        A = Ax * Ay
        return A

def plot_acceptance_map(u, v, acceptance):
    fig, ax = plt.subplots()
    im = ax.imshow(acceptance.T, origin="lower", extent=[u[0], u[-1], v[0], v[-1]], aspect="auto")
    ax.set_xlabel("tan(theta_x)")
    ax.set_ylabel("tan(theta_y)")   
    fig.colorbar(im, label="Acceptance")
    plt.show()

def set_rays(directions, acceptance):
    rays_dirs = []
    rays_weights = []
    rays_pixel_ids = []
    nx, ny = directions.shape[:2]   
    for j in range(ny):
        for i in range(nx):
            dir_vec = directions[i, j]
            rays_dirs.append(dir_vec)
            rays_weights.append(acceptance[i, j])  # weight based on acceptance
            rays_pixel_ids.append(i * ny + j)
    return np.array(rays_dirs), np.array(rays_weights), np.array(rays_pixel_ids)

@numba.njit(inline="always")
def voxel_id(i, j, k, ny, nz):
    return i * ny * nz + j * nz + k

@numba.njit(inline="always")
def voxel_id_vtk(i, j, k, nx, ny):
    return i + nx * (j + ny * k)

@numba.njit
def ray_aabb_intersection(x0, y0, z0, dx, dy, dz,
                          xmin, xmax, ymin, ymax, zmin, zmax):
    tmin = -1e30
    tmax =  1e30

    for o, d, lo, hi in (
        (x0, dx, xmin, xmax),
        (y0, dy, ymin, ymax),
        (z0, dz, zmin, zmax),
    ):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return False, 0.0, 0.0
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False, 0.0, 0.0

    return True, tmin, tmax

@numba.njit
def trace_ray(
    x0, y0, z0,
    dx, dy, dz,
    x_edges, y_edges, z_edges,
    nx, ny, nz,
    rows, cols, data,
    ray_col, weight,
    max_dist=1e30
):
    #Axis-Aligned Bounding Box (AABB) intersection test to find entry and exit points of the ray in the grid
    hit, t0, t1 = ray_aabb_intersection(
        x0, y0, z0, dx, dy, dz,
        x_edges[0], x_edges[-1],
        y_edges[0], y_edges[-1],
        z_edges[0], z_edges[-1],
    )
    if not hit or t1 <= 0:
        # print("hit:", hit, "t0:", t0, "t1:", t1)
        return
    # Point d'entrée dans la grille
    t = max(t0, 0.0)
    x = x0 + t * dx
    y = y0 + t * dy
    z = z0 + t * dz
    # indices initiaux
    i = np.searchsorted(x_edges, x, side="right") - 1
    j = np.searchsorted(y_edges, y, side="right") - 1
    k = np.searchsorted(z_edges, z, side="right") - 1
    # Vérification des indices initiaux (peuvent être hors limites si le point d'entrée est exactement sur une face)
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        # print(f"Initial indices out of bounds: i={i}, j={j}, k={k}")
        return
    # calcul des t pour les prochaines intersections avec les plans de la grille
    if dx == 0.0:
        tx = 1e30
        dtx = 1e30
        sx = 0
    else:
        sx = 1 if dx > 0 else -1
        xb = x_edges[i+1] if dx > 0 else x_edges[i]
        tx = t + (xb - x) / dx
        dtx = abs((x_edges[i+1] - x_edges[i]) / dx)
    if dy == 0.0:
        ty = 1e30
        dty = 1e30
        sy = 0
    else:
        sy = 1 if dy > 0 else -1
        yb = y_edges[j+1] if dy > 0 else y_edges[j]
        ty = t + (yb - y) / dy
        dty = abs((y_edges[j+1] - y_edges[j]) / dy)
    if dz == 0.0:
        tz = 1e30
        dtz = 1e30
        sz = 0
    else:
        sz = 1 if dz > 0 else -1
        zb = z_edges[k+1] if dz > 0 else z_edges[k]
        tz = t + (zb - z) / dz
        dtz = abs((z_edges[k+1] - z_edges[k]) / dz)
    
    # On avance dans la grille en choisissant à chaque étape le plan de grille le plus proche
    while 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and t < min(t1, max_dist):
        i0, j0, k0 = i, j, k

        # Calcul du prochain t d'intersection avec un plan de grille
        if tx <= ty and tx <= tz:
            t_next = tx
            tx += dtx
            i += sx
        elif ty <= tz:
            t_next = ty
            ty += dty
            j += sy
        else:
            t_next = tz
            tz += dtz
            k += sz

        length = max(0.0, min(t_next, t1) - t)
        t = t_next
        vid = voxel_id_vtk(i0, j0, k0, len(x_edges)-1, len(y_edges)-1)
        rows.append(vid)
        cols.append(ray_col)
        data.append(weight * length)

@numba.njit
def accumulate_matrix(
    rays_dirs,
    rays_weights,
    rays_pixel_ids,
    x0, y0, z0,
    x_edges, y_edges, z_edges,
    max_dist
):
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1
    rows = numba.typed.List.empty_list(numba.int64)
    cols = numba.typed.List.empty_list(numba.int64)
    data = numba.typed.List.empty_list(numba.float64)
    for r in range(len(rays_dirs)):
        dx, dy, dz = rays_dirs[r]
        w = rays_weights[r]
        pid = rays_pixel_ids[r]
        trace_ray(
            x0, y0, z0,
            dx, dy, dz,
            x_edges, y_edges, z_edges,
            nx, ny, nz,
            rows, cols, data,
            pid, w,
            max_dist
        )
    return rows, cols, data

def test_single_voxel():
    '''
    Test de la fonction trace_ray pour un cas simple : 
    une grille 1x1x1 de 1m de côté, 
    un rayon parallèle à l'axe x passant par le centre du voxel.
    '''
    x_edges = np.array([0.0, 1.0])
    y_edges = np.array([0.0, 1.0])
    z_edges = np.array([0.0, 1.0])
    # rows = []
    # cols = []
    # data = []
    rows = numba.typed.List.empty_list(numba.int64)
    cols = numba.typed.List.empty_list(numba.int64)
    data = numba.typed.List.empty_list(numba.float64)
    trace_ray(
        x0=-1.0, y0=0.5, z0=0.5,
        dx=1.0, dy=0.0, dz=0.0,
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        nx=1, ny=1, nz=1,
        rows=rows,
        cols=cols,
        data=data,
        ray_col=0,
        weight=1.0,
        max_dist=1e30
    )
    assert len(data) == 1
    assert abs(data[0] - 1.0) < 1e-12
    print("Test passed: ray intersects the voxel with length 1.0")

if __name__ == "__main__":
    
    test_single_voxel()
    
    dir_survey = Path("/Users/raphael/structure/soufriere")
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    vs = 8  # voxel size in m (edge length)
    input_file = dir_voxel / f"topo_voi_vox{vs}m.vts"
    print_file_datetime(input_file)

    args = get_common_args()
    cmn = Common(args)
    tel = cmn.telescope
    tel.compute_angle_matrix()
    x0, y0, z0 = tel.coordinates
    conf_name = "4p"
    azimuth = tel.azimuth_matrix[conf_name]
    zenith = tel.zenith_matrix[conf_name]

    conf = tel.configurations[conf_name]
    size_panel = conf.panels[0].matrix.scintillator.length  # assuming all panels have the same size
    range_tanthetaxy = conf.range_tanthetaxy
    umin, umax, vmin, vmax = range_tanthetaxy[0][0], range_tanthetaxy[0][1], range_tanthetaxy[1][0], range_tanthetaxy[1][1]
    u, v = np.linspace(umin, umax, azimuth.shape[0]), np.linspace(vmin, vmax, azimuth.shape[1])
    delta_z = conf.length_z * 1e-1  # convert mm → cm

    acceptance = angular_acceptance_map(u, v, delta_z=delta_z, Lx=size_panel, Ly=size_panel)
    rays_dirs, rays_weights, rays_pixel_ids = set_rays(tel.directions_matrix[conf_name], acceptance)
    # read source structured grid
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(input_file))
    reader.Update()
    grid = reader.GetOutput()
    voxel_volumes = reader.GetOutput().GetCellData().GetArray("voxel_volume")
    V = np.array(voxel_volumes)
    # assert V.shape[0] == nx * ny * nz, "Voxel volume array size does not match grid dimensions"
    print(f"Voxel volume array shape: {V.shape}")
    pts = grid.GetPoints()
    print(pts.GetNumberOfPoints())
    pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    x_edges = np.unique(pts[:, 0])
    y_edges = np.unique(pts[:, 1])
    z_edges = np.unique(pts[:, 2])
    
    print(f"Grid dimensions: {len(x_edges)-1} x {len(y_edges)-1} x {len(z_edges)-1} voxels")
    max_dist = 1000.0  # max ray length in m
    rows, cols, data = accumulate_matrix(
        rays_dirs,
        rays_weights,
        rays_pixel_ids,
        x0, y0, z0,
        x_edges, y_edges, z_edges,
        max_dist
    )
    n_voxels = (len(x_edges)-1) * (len(y_edges)-1) * (len(z_edges)-1)
    n_pixels = len(rays_dirs)

    M = sp.coo_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(n_voxels, n_pixels)
    ).tocsr()

    # Avoid division by zero in voxel normalization
    V_safe = np.maximum(V, np.finfo(float).eps)
    if np.any(V <= 0):
        n_zero = np.sum(V <= 0)
        print(f"Warning: {n_zero} voxels with zero or negative volume replaced with eps")
    M_normalized = sp.diags(1.0 / V_safe) @ M
    f_out = dir_voxel / f"voxel_ray_matrix_{tel.name}_{conf_name}_vox{vs}m.npz"
    np.savez_compressed(f_out, M=M_normalized, x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    print(f"Saved voxel-ray matrix with shape {M_normalized.shape} and {M_normalized.nnz} non-zero elements")

    # Save intercepted voxels with pixel assignments for ParaView visualization
    reader_out = vtk.vtkXMLStructuredGridReader()
    reader_out.SetFileName(str(input_file))
    reader_out.Update()
    grid_out = reader_out.GetOutput()
    
    # Keep the original voxel_volume array
    grid_out.GetCellData().AddArray(voxel_volumes)
    
    # Build cell array: for each voxel, store the primary pixel that intercepts it (highest weight)
    M_csr = M.tocsr()
    n_cells = M_csr.shape[0]
    primary_pixel = vtk.vtkIntArray()
    primary_pixel.SetName("primary_pixel")
    primary_pixel.SetNumberOfComponents(1)
    primary_pixel.SetNumberOfTuples(n_cells)
    
    intercepted_weight = vtk.vtkFloatArray()
    intercepted_weight.SetName("intercepted_weight")
    intercepted_weight.SetNumberOfComponents(1)
    intercepted_weight.SetNumberOfTuples(n_cells)
    
    n_intercepting_pixels = vtk.vtkIntArray()
    n_intercepting_pixels.SetName("num_intercepting_pixels")
    n_intercepting_pixels.SetNumberOfComponents(1)
    n_intercepting_pixels.SetNumberOfTuples(n_cells)
    
    for vid in tqdm(range(n_cells), desc="Assigning primary pixels to voxels"):
        row = M_csr.getrow(vid)
        if row.nnz > 0:
            # Find pixel with max weight for this voxel
            col_indices = row.indices
            col_data = row.data
            max_idx = np.argmax(col_data)
            primary_pixel.SetValue(vid, int(col_indices[max_idx]))
            intercepted_weight.SetValue(vid, float(col_data[max_idx]))
            n_intercepting_pixels.SetValue(vid, int(row.nnz))
        else:
            primary_pixel.SetValue(vid, -1)
            intercepted_weight.SetValue(vid, 0.0)
            n_intercepting_pixels.SetValue(vid, 0)
    
    grid_out.GetCellData().AddArray(primary_pixel)
    grid_out.GetCellData().AddArray(intercepted_weight)
    grid_out.GetCellData().AddArray(n_intercepting_pixels)
    
    f_vts = dir_voxel / f"voxel_intercepted_{tel.name}_{conf_name}_vox{vs}m.vts"
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(f_vts))
    writer.SetInputData(grid_out)
    writer.Write()
    print(f"Saved intercepted voxels visualization: {f_vts}")


