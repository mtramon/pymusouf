#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
from pathlib import Path
import vtk
from tqdm import tqdm
#package module(s)
from config import STRUCT_DIR
from telescope import DICT_TEL
from survey import CURRENT_SURVEY

def sph_to_dir(az, ze):
    return np.array([
        np.sin(ze) * np.cos(az),
        np.sin(ze) * np.sin(az),
        np.cos(ze)
    ])

if __name__=="__main__":


    survey_name = CURRENT_SURVEY.name   
    dir_survey = STRUCT_DIR / survey_name

    dtel = CURRENT_SURVEY.telescopes
    # dtel = {"OM":DICT_TEL["OM"]}
   
    for tel_name, tel in tqdm(dtel.items(), desc="Raypath"):
        for conf_name, conf in tel.configurations.items():
            conf = tel.configurations[conf_name] #if conf_name in tel.configurations.keys() else
            dir_acq = dir_survey / "telescope" / tel_name / "acqvars"/f"az{tel.azimuth}ze{tel.zenith}"
            file_acqvar = dir_acq / f"acqVars_{conf_name[:2]}.npz" 
            if not file_acqvar.exists(): 
                print(f"{file_acqvar} does not exist")
                continue
            acqvar = np.load(file_acqvar)
            thickness = acqvar["apparentThickness"]
            tel_pos = tel.coordinates 
            tel.compute_angular_coordinates()
            azimuth = tel.azimuth_matrix[conf_name]
            zenith = tel.zenith_matrix[conf_name]
            delta_az, delta_ze = np.median(np.diff(azimuth.ravel())),  np.median(np.diff(zenith.ravel()))             
            Lmin = 1.
            Lmax = 1200.0
            nx, ny = azimuth.shape
            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()
            ugrid = vtk.vtkUnstructuredGrid()
            # --- CellData arrays ---
            pid_arr = vtk.vtkIntArray()
            pid_arr.SetName("pid")
            az_arr = vtk.vtkFloatArray()
            az_arr.SetName("azimuth")
            ze_arr = vtk.vtkFloatArray()
            ze_arr.SetName("zenith")
            len_arr = vtk.vtkFloatArray()
            len_arr.SetName("ray_length")
            # op_arr = vtk.vtkFloatArray()
            # op_arr.SetName("opacity")
            # --- Génération des lignes ---
            pid = 0
            point_id = 0
            # demi-ouvertures angulaires du pixel
            daz = delta_az
            dze = delta_ze
            
            def add_quad(a, b, c, d):
                q = vtk.vtkQuad()
                q.GetPointIds().SetId(0, a)
                q.GetPointIds().SetId(1, b)
                q.GetPointIds().SetId(2, c)
                q.GetPointIds().SetId(3, d)
                polys.InsertNextCell(q)
            
            for j in range(ny):
                for i in range(nx):
                    az = azimuth[i, j]
                    ze = zenith[i, j]
                    # 4 coins angulaires
                    corners = [
                        (az - daz/2, ze - dze/2),
                        (az + daz/2, ze - dze/2),
                        (az + daz/2, ze + dze/2),
                        (az - daz/2, ze + dze/2),
                    ]
                    ids_near = []
                    ids_far  = []
                    # 8 sommets
                    k=0
                    for L, ids in [(Lmin, ids_near), (Lmax, ids_far)]:
                        for azc, zec in corners:
                            p = tel_pos + L * sph_to_dir(azc, zec)
                            points.InsertNextPoint(p)
                            ids.append(point_id)
                            point_id += 1
                            k+=1
                    # proche
                    add_quad(ids_near[0], ids_near[1], ids_near[2], ids_near[3])
                    # # lointaine (ordre inversé)
                    add_quad(ids_far[3], ids_far[2], ids_far[1], ids_far[0])
                    for k in range(4):
                        add_quad(
                            ids_near[k],
                            ids_near[(k+1)%4],
                            ids_far[(k+1)%4],
                            ids_far[k]
                        )
                    l = thickness[i,j]
                    length= l if ~np.isnan(l) else 0
                    # # --- CellData ---
                    for _ in range(6):
                        pid_arr.InsertNextValue(pid)
                        az_arr.InsertNextValue(az)
                        ze_arr.InsertNextValue(ze)
                        len_arr.InsertNextValue(length)
                    pid += 1

            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(polys)
            cd = poly.GetCellData()
            cd.AddArray(pid_arr)
            cd.AddArray(az_arr)
            cd.AddArray(ze_arr)
            cd.AddArray(len_arr)
            cd.SetActiveScalars("len_arr")

            writer = vtk.vtkXMLPolyDataWriter()
            file_out = dir_acq / f"raypath_{tel_name}_{conf_name[:2]}.vtp"
            writer.SetFileName(file_out)
            name = f"{tel_name}" if tel_name != "SNJ" else f"{tel_name}_{conf_name[:2]}"
            writer.SetInputData(poly)

            writer.Write()
            print(f"Save {file_out}")