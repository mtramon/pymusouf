import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path


def load_density_model(file):
    """
    Load a density array from a model file (.vts or .npz).

    The file must contain an array named 'density'.
    """
    file = Path(file)
    if file.suffix == ".vts":
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(file))
        reader.Update()
        grid = reader.GetOutput()
        arr_vtk = grid.GetCellData().GetArray("density")
        if arr_vtk is None:
            raise ValueError(f"No 'density' array found in {file}")
        # density = vtk_to_numpy(arr_vtk)
        density = np.array(arr_vtk)
    elif file.suffix == ".npz":
        data = np.load(file)
        if "density" not in data:
            raise ValueError(f"No 'density' array found in {file}")
        density = data["density"]
    else:
        raise ValueError(f"Unsupported model format: {file.suffix}")

    if max(density) < 1e3: density *= 1e3 #conversion from g/cm^3 to kg/m^3

    return density