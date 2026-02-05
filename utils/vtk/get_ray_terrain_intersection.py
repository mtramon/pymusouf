import vtk
from pathlib import Path 
from datetime import datetime

dir_main = Path("/Users/raphael/pymusouf/files/survey/soufriere/") 
dir_dem = dir_main / "dem"
# Terrain
terrain_reader = vtk.vtkXMLUnstructuredGridReader()
file_terrain = dir_dem / "soufriere_dome_volume_5m.vtu"
print(f"Load {file_terrain} -- {datetime.fromtimestamp(file_terrain.stat().st_mtime)}")

terrain_reader.SetFileName(str(file_terrain))
terrain_reader.Update()
# Surface fermée du terrain
terrain_surface = vtk.vtkDataSetSurfaceFilter()
terrain_surface.SetInputConnection(terrain_reader.GetOutputPort())
terrain_surface.Update()

# Sélecteur spatial
enclosed = vtk.vtkSelectEnclosedPoints()
enclosed.SetSurfaceData(terrain_surface.GetOutput())
enclosed.SetTolerance(1e-6)

# Détecteur
dir_tel = dir_main / "telescope"
detectors = {
        "SB": dir_tel / "SB/acqvars/az40.0ze79.0/raypath_SB_3p.vtu",
        "SNJ_3p" : dir_tel / "SNJ/acqvars/az18.0ze74.9/raypath_SNJ_3p.vtu",
       "SNJ_4p":dir_tel / "SNJ/acqvars/az18.0ze74.9/raypath_SNJ_4p.vtu",
        "BR": dir_tel / "BR/acqvars/az297ze80/raypath_BR_3p.vtu",
    
        "OM": dir_tel / "OM/acqvars/az192ze76.6/raypath_OM_3p.vtu",
    }
for k, v in detectors.items() : 
    det_reader = vtk.vtkXMLUnstructuredGridReader()
    det_reader.SetFileName(str(v))
    det_reader.Update()

    print(f"Load {v} -- {datetime.fromtimestamp(v.stat().st_mtime)}")

    # Test d'inclusion des points
    enclosed.SetInputData(det_reader.GetOutput())
    enclosed.Update()

    # Extraction des cellules internes
    inside_cells = vtk.vtkThreshold()
    inside_cells.SetInputConnection(enclosed.GetOutputPort())
    inside_cells.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "SelectedPoints"
    )
    inside_cells.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
    inside_cells.SetUpperThreshold(0.5)
    inside_cells.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    file_out = v.parent / f"intersect_{k}_{file_terrain.stem}.vtu"
    writer.SetFileName(file_out)
    writer.SetInputConnection(inside_cells.GetOutputPort())
    writer.Write()
    print(f"Save {file_out}")