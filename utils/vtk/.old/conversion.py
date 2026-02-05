#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np 
from pathlib import Path
import vtk
from vtk.util import numpy_support 

def convert_3d_npy_to_vti(arr:np.ndarray, file:str|Path="density_model.vti"):
    '''
    Conversion of numpy array to vti for Paraview visualisation (based on vtk)
    
    arr: numpy array of 3-d shape (x, y, z)
    file: path to vti output file 
    '''
    vtk_data = numpy_support.numpy_to_vtk(arr.ravel(order='F'), deep=True)
    image = vtk.vtkImageData()
    shp = arr.shape
    image.SetDimensions(shp[0], shp[1], shp[2])
    image.GetPointData().SetScalars(vtk_data)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(image)
    writer.SetDataModeToBinary()   # important !
    writer.EncodeAppendedDataOff() # évite "raw", force base64
    writer.Write()
    # print(f"Save {file}")
    
    
def convert_4d_npy_to_vtm(arr, file:str|Path="density_model.vtm", spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Sauvegarde un tableau numpy 4D (t, x, y, z) dans un seul fichier .vtm (multi-block) compatible ParaView.
    
    Paramètres
    ----------
    data : np.ndarray
        Tableau 4D de forme (t, x, y, z)
    output_filename : str
        Nom du fichier .vtm de sortie
    spacing : tuple de float
        Espacement entre les points dans chaque direction (dx, dy, dz)
    origin : tuple de float
        Coordonnées de l’origine de la grille
    """
    assert arr.ndim == 4, "Le tableau doit être 4D (t, x, y, z)"

    nt, nx, ny, nz = arr.shape
    spacing = tuple(map(float, spacing))
    origin = tuple(map(float, origin))

    # Création du multiblock
    multiblock = vtk.vtkMultiBlockDataSet()

    for i in range(nt):
        # Création de l'objet vtkImageData pour ce pas de temps
        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)

        # Conversion NumPy → VTK
        flat_data = arr[i].ravel(order='F')  # Fortran order important pour VTK
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName("scalar")

        # Ajout des scalaires à l'image
        img.GetPointData().SetScalars(vtk_array)

        # Ajout de l'image comme bloc du multiblock
        multiblock.SetBlock(i, img)
        multiblock.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), f"k_{i}")

    # Écriture du fichier .vtm
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(multiblock)
    writer.Write()
    print(f"Save {file}")

def convert_4d_npy_to_vti_serie(arr: np.ndarray, output_prefix: str = "density", output_dir: str | Path = ".", create_pvd: bool = True):
    """
    Convertit un tableau 4D (t, x, y, z) en une série de fichiers .vti
    + un fichier .pvd optionnel pour animation temporelle dans ParaView.

    Paramètres
    ----------
    data : np.ndarray
        Tableau 4D (t, x, y, z)
    output_prefix : str
        Préfixe des fichiers .vti générés (ex: 'density' -> density_0000.vti)
    output_dir : str | Path
        Répertoire de sortie
    create_pvd : bool
        Si True, crée un fichier .pvd regroupant tous les pas de temps
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nt = arr.shape[0]
    vti_files = []

    for i in range(nt):
        filename = output_dir / f"{output_prefix}_{i:04d}.vti"
        if not np.all(arr[i]==0):  
            convert_3d_npy_to_vti(arr[i], filename)
            vti_files.append(filename.name)
            print(f"Save {filename}")
