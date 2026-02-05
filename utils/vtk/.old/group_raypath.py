import vtk
from pathlib import Path

dir_tel = Path("/Users/raphael/pymusouf/files/survey/soufriere/telescope/") 

dict_det_ug = {
    "SB": str(dir_tel / "SB/acqvars/az40.0ze79.0/raypath_SB_3p.vtu"),
    "SNJ_3p": str(dir_tel / "SNJ/acqvars/az18.0ze74.9/raypath_SNJ_3p.vtu"),
    "SNJ_4p": str(dir_tel / "SNJ/acqvars/az18.0ze74.9/raypath_SNJ_4p.vtu"),
    "BR" :  str(dir_tel / "BR/acqvars/az297ze80/raypath_BR_3p.vtu"),
    "OM" : str(dir_tel / "OM/acqvars/az192ze76.6/raypath_OM_3p.vtu")
}

# ------------------------------------------------------------
# 1. Lecture des détecteurs (UnstructuredGrid)
# ------------------------------------------------------------

readers = []

for name, filename in dict_det_ug.items():
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    readers.append((reader, name))

# ------------------------------------------------------------
# 2. Création du MultiBlockDataSet
# ------------------------------------------------------------

mb = vtk.vtkMultiBlockDataSet()
mb.SetNumberOfBlocks(len(readers))

for i, (reader, name) in enumerate(readers):
    mb.SetBlock(i, reader.GetOutput())
    mb.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), name)

# ------------------------------------------------------------
# 3. Écriture dans un seul fichier VTK
# ------------------------------------------------------------

file_out = dir_tel / f"raypath_{'_'.join(list(dict_det_ug.keys()))}.vtm"
writer = vtk.vtkXMLMultiBlockDataWriter()
writer.SetFileName(file_out)
writer.SetInputData(mb)
writer.Write()
print(f"Save {file_out}")




