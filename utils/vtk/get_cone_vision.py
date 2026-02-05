#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import vtk
#package module(s)
from survey import CURRENT_SURVEY

if __name__=="__main__":
    souf_survey = CURRENT_SURVEY
    tel_name = "SNJ"
    tel = souf_survey.telescope[tel_name]
    conf_name = "3p1"
    conf = tel.configurations[conf_name] #if conf_name in tel.configurations.keys() else
    dir_path = souf_survey.path
    dir_acc = dir_path / "telescope" / tel_name / "acqvars"/f"az{tel.azimuth}ze{tel.zenith}"
    rmax = 1000
    tel.get_ray_paths(front_panel=conf.panels[0], 
                            rear_panel=conf.panels[-1], 
                            rmax=rmax)
    ray_matrix = tel.rays
    nray = tel.rays.shape[0]
    s = int(np.sqrt(nray))

    directions = ray_matrix[:,1] - ray_matrix[:,0]
    directions /= np.linalg.norm(directions, axis=-1)[:,np.newaxis]
    mean_dir = np.mean(directions, axis=0)
    directions = directions.reshape((s,s,3))
    # directions shape (s, s, 3)
    d00 = directions[0, 0]
    d30 = directions[s-1, 0]
    d33 = directions[s-1, s-1]
    d03 = directions[0, s-1]
    dirs = [d00, d30, d33, d03]
    points_cone = vtk.vtkPoints()
    A = tel.coordinates # Apex
    points_cone.InsertNextPoint(A)
    # Base carrée
    for d in dirs:
        B = A + rmax * d
        points_cone.InsertNextPoint(B)
    polys = vtk.vtkCellArray()
    for i in range(1, 5):
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, 0)          # apex
        tri.GetPointIds().SetId(1, i)
        tri.GetPointIds().SetId(2, 1 + i % 4)
        polys.InsertNextCell(tri)
    quad = vtk.vtkQuad()
    quad.GetPointIds().SetId(0, 1)
    quad.GetPointIds().SetId(1, 2)
    quad.GetPointIds().SetId(2, 3)
    quad.GetPointIds().SetId(3, 4)
    polys.InsertNextCell(quad)
    pyramid = vtk.vtkPolyData()
    pyramid.SetPoints(points_cone)
    pyramid.SetPolys(polys)
    # writer = vtk.vtkXMLPolyDataWriter()
    # file_out = dir_acc / f"vision_cone_{tel_name}_{conf_name}.vtp"
    # writer.SetFileName(file_out)
    # writer.SetInputData(pyramid)
    # writer.Write()
    # print(f"Save {file_out}")
   
    ###ALL RAYS
    pid = 0
    points_rays = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for i in range(nray):
        p1,p2 = ray_matrix[i]  # vecteur unitaire
        points_rays.InsertNextPoint(p1)
        points_rays.InsertNextPoint(p2)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, pid)
        line.GetPointIds().SetId(1, pid + 1)
        lines.InsertNextCell(line)
        pid += 2
    poly_rays = vtk.vtkPolyData()
    poly_rays.SetPoints(points_rays)
    poly_rays.SetLines(lines)
    # Écriture fichier
    writer = vtk.vtkXMLPolyDataWriter()
    file_out = dir_acc / f"raypaths_{tel_name}_{conf_name}.vtp"
    writer.SetFileName(file_out)
    writer.SetInputData(pyramid)
    writer.SetInputData(poly_rays)
    writer.Write()
    print(f"Save {file_out}")