from paraview.simple import *
from pathlib import Path

dir_tel = Path("/Users/raphael/pymusouf/files/survey/soufriere/telescope/") 
file_raypath = dir_tel / "raypath_SB_SNJ_3p_SNJ_4p_BR_OM.vtm"

# ------------------------------------------------------------
# Détecteurs : fichier, nom, couleur
# ------------------------------------------------------------

detectors = [
    {
        "file": dir_tel / "SB/acqvars/az40.0ze79.0/raypath_SB_3p.vtu",
        "name": "SB",
        "color": [0.0, 0.0, 1.0],  
    },
    {
        "file": dir_tel / "SNJ/acqvars/az18.0ze74.9/raypath_SNJ_3p.vtu",
        "name": "SNJ_3p",
        "color": [1.0, 1.0, 0.0],  # jaune
    },
    {
        "file": dir_tel / "SNJ/acqvars/az18.0ze74.9/raypath_SNJ_4p.vtu",
        "name": "SNJ_4p",
        "color": [1.0, 1.0, 0.0],  # jaune
    },
    {
        "file": dir_tel / "BR/acqvars/az297ze80/raypath_BR_3p.vtu",
        "name": "BR",
        "color":  [1.0, 0.0, 0.0],  # rouge
    },
    {
        "file": dir_tel / "OM/acqvars/az192ze76.6/raypath_OM_3p.vtu",
        "name": "OM",
        "color":   [0.8, 0.5, 0.0],  # orange
    },
]


view = GetActiveViewOrCreate("RenderView")

sources = []

# ------------------------------------------------------------
# Chargement + affichage
# ------------------------------------------------------------

for det in detectors:
    reader = XMLUnstructuredGridReader(
        FileName=[str(det["file"])]
    )
    reader.UpdatePipeline()

    # Nom explicite dans le Pipeline Browser
    RenameSource(det["name"], reader)

    display = Show(reader, view)
    display.SetRepresentationType("Surface")
    display.ColorArrayName = [None, '']  # Solid Color
    display.DiffuseColor = det["color"]
    display.Opacity = .75
    display.Specular = 0.1

    sources.append(reader)

# ------------------------------------------------------------
# Vue
# ------------------------------------------------------------

view.ResetCamera()
Render()