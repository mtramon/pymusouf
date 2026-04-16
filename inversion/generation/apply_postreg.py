#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import h5py
import glob
import numpy as np
from pathlib import Path
import pickle
import re
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm
import vtk 
from vtk.util import numpy_support
import torch

# package module(s)
from config import STRUCT_DIR
from inversion.model import load_density_model
from inversion.tv import InversionTV
from inversion.voxelgrid import load_voxel_grid, get_coordinates
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
from training import UNet3D

# ----------------------------------------------------------------------
# Fonctions de chargement des fichiers VTK / NPZ
# ----------------------------------------------------------------------
def load_vtk_density(filename, array_name='density_reg'):
    """Charge un fichier VTK structuré et retourne le champ de densité 3D et le masque."""
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    grid = reader.GetOutput()

    # Récupérer les dimensions
    dims = grid.GetDimensions()  # (nx_pts, ny_pts, nz_pts)
    nvx, nvy, nvz = dims[0]-1, dims[1]-1, dims[2]-1

    # Lire le champ de densité
    arr = grid.GetCellData().GetArray(array_name)
    if arr is None:
        raise ValueError(f"Le fichier VTK ne contient pas l'array '{array_name}'")
    density = numpy_support.vtk_to_numpy(arr).reshape((nvx, nvy, nvz), order='F')
    mask = (density > 0)#.astype(np.uint8)
    # Lire le masque (voxel_volume > 0) s'il existe, sinon le créer à partir de la densité non nulle
    vol_arr = grid.GetCellData().GetArray("voxel_volume")
    if vol_arr is not None:
        volume = numpy_support.vtk_to_numpy(vol_arr)
        mask = volume.reshape((nvx, nvy, nvz), order='F') > 0
    # else:
        # Si pas de volume, on utilise la densité non nulle comme masque (approximatif)

    return density, mask, (nvx, nvy, nvz)

def load_npz_data(filename, key='X'):
    """Charge un fichier NPZ contenant un champ de densité et un masque."""
    data = np.load(filename)
    # On suppose que le fichier contient 'X' (postérieur) et 'mask', ou bien 'density' et 'mask'
    if 'X' in data:
        density = data['X']
    elif 'density_reg' in data:
        density = data['density_reg']
    else:
        raise KeyError("Le fichier NPZ doit contenir 'X' ou 'density_reg'")
    if 'mask' in data:
        mask = data['mask']
    else:
        mask = density > 0
    # Vérifier si density est 1D ou 3D
    if density.ndim == 1:
        # Nécessite les dimensions
        if "shape" in data:
            nvx, nvy, nvz = data['shape']
            density = density.reshape((nvx, nvy, nvz), order='F')
            mask = mask.reshape((nvx, nvy, nvz), order='F')
        else:
            raise ValueError("Fichier NPZ avec density 1D nécessite nx, ny, nz")
    else:
        nvx, nvy, nvz = density.shape
    return density, mask, (nvx, nvy, nvz)

# ----------------------------------------------------------------------
# Application du modèle
# ----------------------------------------------------------------------
def apply_model(model, volume, mask, mean, std, device):
    """
    Applique le modèle à un volume 3D (numpy array) après normalisation.
    Retourne le volume corrigé (numpy array) dénormalisé.
    """
    model.eval()
    # Ajouter dimensions batch et canal
    vol_tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    # Normalisation
    vol_tensor = (vol_tensor - mean) / (std + 1e-8)
    with torch.no_grad():
        out_tensor = model(vol_tensor)
    # Dénormalisation
    out_tensor = out_tensor * std + mean
    # Revenir à numpy
    out = out_tensor.squeeze().cpu().numpy()
    # Appliquer le masque (optionnel)
    out[~mask] = volume[~mask]  # ou 0, selon choix
    return out

def save_vtk_density(template_file, output_file, density, mask, array_name='density_post'):
    """
    Sauvegarde le champ de densité dans un fichier VTK en utilisant un fichier template
    pour récupérer la grille.
    """
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(template_file))
    reader.Update()
    grid = vtk.vtkStructuredGrid()
    # grid.DeepCopy(reader.GetOutput())
    grid.ShallowCopy(reader.GetOutput())


    # Convertir le tableau numpy en vtkArray$
    mask_flat = mask.ravel(order='F').astype(np.uint8)
    density_flat = density.ravel(order='F')
    density_flat[mask_flat] = np.clip(density_flat[mask_flat], 1800, 2700)
    vtk_array = numpy_support.numpy_to_vtk(density_flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName(array_name)
    grid.GetCellData().AddArray(vtk_array)

    # Ajouter éventuellement le masque (optionnel)
    vtk_mask = numpy_support.numpy_to_vtk(mask_flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_mask.SetName("mask")
    grid.GetCellData().AddArray(vtk_mask)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(output_file))
    writer.SetInputData(grid)
    writer.Write()
    print(f"Résultat sauvegardé dans {output_file}")


if __name__ =="__main__":

    survey_name = CURRENT_SURVEY.name
    dir_survey = STRUCT_DIR / survey_name
    dirs = {
        "survey": dir_survey,
        "dem": dir_survey / "dem",
        "voxel": dir_survey / "voxel",
        "model": dir_survey / "model",
        "tel": dir_survey / "telescope",
    }
    dirs["post"] = dirs["model"] / "postreg"
    input_data = dirs["post"] / "training_data.npz"
    dirs["train"] = dirs["post"] / "training"
    dirs["train"].mkdir(parents=True, exist_ok=True)

    input_file = dirs["post"] / "ElecCond_topo_voi_vox32m_reg.npz"
    print_file_datetime(input_file)

    output_file = dirs["post"] / "ElecCond_topo_voi_vox32m_post.vts"
    parser = argparse.ArgumentParser(description="Application post‑régularisation par U‑Net")
    parser.add_argument('--checkpoint', type=str, default=str(dirs["train"] / "best_model.pth"),
                        help="Fichier .pth contenant les poids du modèle entraîné")
    parser.add_argument('--stats', type=str, default=str(dirs["train"]/"norm_stats.npz"),
                        help="Fichier .npz contenant mean_train, std_train (et éventuellement masque)")
    parser.add_argument('--input', type=str, default=str(input_file),
                        help="Fichier d'entrée (VTK ou NPZ) contenant le modèle postérieur à corriger")
    parser.add_argument('--output', type=str, default=str(output_file),
                        help="Fichier de sortie (VTK de préférence)")
    parser.add_argument('--gpu', action='store_true', help="Utiliser GPU si disponible")
    args = parser.parse_args()
    

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Charger les statistiques de normalisation
    stats = np.load(args.stats)
    mean_train = stats['mean'].item()
    std_train = stats['std'].item()
    print(f"Stats de normalisation : mean={mean_train:.4f}, std={std_train:.4f}")

    # Charger le modèle
    # Déterminer les dimensions à partir du checkpoint (ou utiliser des valeurs par défaut)
    # Ici on suppose que le modèle a été entraîné avec max_levels=3 et base_features=16
    model = UNet3D(in_channels=1, out_channels=1, base_features=16, max_levels=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    print_file_datetime(args.checkpoint)
    print("Modèle chargé.")

    # Charger les données d'entrée
    input_path = Path(args.input)
    if input_path.suffix == '.vts':
        density_input, mask, (nx, ny, nz) = load_vtk_density(input_path)
    elif input_path.suffix == '.npz':
        density_input, mask, (nx, ny, nz) = load_npz_data(input_path)
    else:
        raise ValueError("Format de fichier non supporté. Utilisez .vts ou .npz")
    print(f"Volume chargé : {density_input.shape} (nx, ny, nz) = ({nx}, {ny}, {nz})")
    print(f"mask : {np.count_nonzero(mask)}")
    # Appliquer le modèle
    density_post = apply_model(model, density_input, mask, mean_train, std_train, device)
    print(density_post.shape)
    print(np.min(density_post), np.max(density_post), np.mean(density_post), np.median(density_post))
    # Sauvegarder le résultat
    output_path = Path(args.output) 
    input_vtk = input_file.parent / f"{input_file.stem}.vts"
    if output_path.suffix == '.vts':
        # On utilise le fichier d'entrée comme template (pour la grille)
        save_vtk_density(input_vtk, output_path, density_post, mask, 'density_post')
    else:
        # Sinon sauvegarde en NPZ
        np.savez_compressed(output_path, density_post=density_post, mask=mask)
        print(f"Résultat sauvegardé dans {output_path}")
