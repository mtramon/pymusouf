import numpy as np
from scipy.interpolate import RegularGridInterpolator
import vtk

def build_interpolator_topography(file):
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(file)
        reader.Update()
        surf = reader.GetOutput()
        points = surf.GetPoints()
        xmin, xmax, ymin, ymax, zmin, zmax = surf.GetExtent()
        nx, ny = xmax-xmin+1, ymax-ymin+1
        Xtopo = np.zeros((nx, ny))
        Ytopo = np.zeros((nx, ny))
        Ztopo = np.zeros((nx, ny))
        idx = 0
        for j in range(ny):
            for i in range(nx):
                _x, _y, _z = points.GetPoint(idx)
                Xtopo[i, j] = _x
                Ytopo[i, j] = _y
                Ztopo[i, j] = _z
                idx += 1
        x_unique, y_unique = np.unique(Xtopo.ravel()), np.unique(Ytopo.ravel())
        # j_sort = np.argsort(y_unique)
        # y_unique = y_unique[j_sort]
        # Ztopo = Ztopo[:,j_sort]
        interp = RegularGridInterpolator(
            (x_unique, y_unique),
            Ztopo.T,  
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        return interp

def interpolate_topography(interp, x, y):
        pts = np.stack((x, y))
        pts = np.moveaxis(pts, 0, -1)
        z_interp = interp(pts)
        z_interp = z_interp.T
        '''
        # ----------------------------
        # Création du Structured Grid VTK
        # ----------------------------
        points = vtk.vtkPoints()
        nx, ny = len(x), len(y)
        points.SetNumberOfPoints(nx * ny)
        idx = 0
        for j in range(ny):
            for i in range(nx):
                points.SetPoint(idx, x[i, j], y[i, j], z_interp[i, j])
                idx += 1
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, 1)
        grid.SetPoints(points)
        # ----------------------------
        # Écriture .vts
        # ----------------------------
        writer = vtk.vtkXMLStructuredGridWriter()
        fout ="test_interp.vts"
        writer.SetFileName(fout)
        writer.SetInputData(grid)
        writer.Write()
        print(f"Topographie interpolée sauvegardée : {fout}")
        '''
        return z_interp