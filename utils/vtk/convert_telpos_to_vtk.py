#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numba
import numpy as np
import vtk
from tqdm import tqdm
import scipy.sparse as sp
import sys
import vtk
from vtk.util import numpy_support

# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
from telescope import DICT_TEL

@numba.njit
def adjust_detectors_height(
    det_positions,
    x_edges, y_edges, z_edges,
    mask_voxel,
    offset=1.0
):
    """
    Ajuste la position Z de plusieurs détecteurs si sous la topographie voxelisée.

    Paramètres
    ----------
    det_positions : (N,3)
    offset : hauteur minimale au-dessus de la surface (m)

    Retour
    ------
    new_positions : (N,3)
    moved_mask : (N,) bool
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1
    N = det_positions.shape[0]
    new_positions = det_positions.copy()
    moved_mask = np.zeros(N, dtype=np.uint8)
    for d in range(N):
        x0 = det_positions[d, 0]
        y0 = det_positions[d, 1]
        z0 = det_positions[d, 2]
        i = -1
        j = -1
        for ix in range(nx):
            if x_edges[ix] <= x0 < x_edges[ix+1]:
                i = ix
                break
        for iy in range(ny):
            if y_edges[iy] <= y0 < y_edges[iy+1]:
                j = iy
                break
        if i == -1 or j == -1:
            continue
        top_z = -1e30
        found = False
        for k in range(nz):
            vid = i + nx * (j + ny * k)
            if mask_voxel[vid] == 1:
                top_z = z_edges[k+1]
                found = True
        if not found:
            continue
        surface_z = top_z
        znew =surface_z + offset
        if (z0 < znew):
            new_positions[d, 2] = znew
            moved_mask[d] = 1
    return new_positions, moved_mask

def save_detectors_vtk(
    old_positions,
    new_positions,
    moved_mask,
    filename
):
    """
    Sauvegarde positions anciennes et nouvelles des détecteurs
    en vtkPolyData.
    """

    n = old_positions.shape[0]
    # Points = nouvelles positions
    points = vtk.vtkPoints()
    points.SetData(
        numpy_support.numpy_to_vtk(new_positions.astype(np.float64))
    )

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # --- Cellules vertices ---
    vertices = vtk.vtkCellArray()
    for i in range(n):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)

    polydata.SetVerts(vertices)

    # --- Scalar : moved ---
    moved_vtk = numpy_support.numpy_to_vtk(
        moved_mask.astype(np.float64)
    )
    moved_vtk.SetName("adjusted_position")
    polydata.GetPointData().AddArray(moved_vtk)

    # --- Vecteur déplacement ---
    displacement = new_positions - old_positions
    disp_vtk = numpy_support.numpy_to_vtk(
        displacement.astype(np.float64)
    )
    disp_vtk.SetName("displacement")
    polydata.GetPointData().AddArray(disp_vtk)

    # --- Ancienne position ---
    old_vtk = numpy_support.numpy_to_vtk(
        old_positions.astype(np.float64)
    )
    old_vtk.SetName("old_position")
    polydata.GetPointData().AddArray(old_vtk)

    # Écriture
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


if __name__ == "__main__":
        
    survey_name = CURRENT_SURVEY.name   
    print(f"Processing survey: {survey_name}")
    print(f"Survey structure directory: {STRUCT_DIR / survey_name}")
    dir_survey = STRUCT_DIR / survey_name
    dir_tel = dir_survey / "telescope"
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    vs = sys.argv[1] if len(sys.argv) >1 else 8  # voxel size in m (edge length)
    input_file = dir_voxel / f"topo_voi_vox{vs}m.vts"
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
    print(f"Voxel volume array shape: {voxel_volume.shape}")
    pts = grid.GetPoints()
    pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    x_edges = np.unique(pts[:, 0])
    y_edges = np.unique(pts[:, 1])
    z_edges = np.unique(pts[:, 2])
    print(f"Grid dimensions: {len(x_edges)-1} x {len(y_edges)-1} x {len(z_edges)-1} voxels")

    dtel = CURRENT_SURVEY.telescopes
    det_positions = np.array([tel.coordinates for _,tel in dtel.items()])

    new_pos, moved = adjust_detectors_height(
        det_positions,
        x_edges, y_edges, z_edges,
        mask_voxel,
        offset=1.0
    )
    fout_vtk = dir_tel / f"telescopes_adjusted_positions_vox{vs}m.vtp"
    save_detectors_vtk(
        det_positions,
        new_pos,
        moved,
        fout_vtk
    )

    print(f"Saved {fout_vtk}")