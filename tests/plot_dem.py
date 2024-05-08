# -*- coding: utf-8 -*-
#!/usr/bin/env python3

#%%
import numpy as np
from pathlib import Path
import time
from scipy.io import loadmat
import pickle
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

#personal modules
from raypath import RayPath
from survey import CURRENT_SURVEY, DICT_SURVEY

if __name__ == "__main__":

    survey = CURRENT_SURVEY
    main_path = Path(__file__).parents[1]
  
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(projection='3d')
    X1, Y1, Z1 = survey.surface_grid
    ax.plot_surface(X1, Y1, Z1, alpha=0.2, color='blue', label='structure') 

    # ax.legend()
    
    rmax = 1500

    ###thickness color scale
    import palettable
    import matplotlib.colors as cm
    import matplotlib.colors as colors
    #from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
    cmap_thick = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    L_min, L_max= 0, rmax 
    range_thick = np.linspace(0,rmax, 100) 
    norm_r = cm.Normalize(vmin=L_min, vmax=L_max)(range_thick)
    color_scale_thick =  cmap_thick(norm_r)
    

    for name, tel in survey.telescopes.items():
        x,y,z = tel.utm
        ax.scatter(x,y,z, color=tel.color, label=name)
        raypath = RayPath(telescope=tel,
                    surface_grid=survey.surface_grid)
        pickle_file = survey.path/ 'telescope'/ tel.name/'raypath'/f'az{tel.azimuth}_elev{tel.elevation}'/'raypath'
        raypath( pickle_file , max_range=rmax) 
        thick = raypath.raypath['3p1']['thickness'].flatten()
        arg_col =  [np.argmin(abs(range_thick-v))for v in thick]   
        color_values = color_scale_thick[arg_col] 
        mask = (np.isnan(thick)) 
        tel.plot_ray_values(ax, color_values=color_values, front_panel=tel.panels[0], rear_panel=tel.panels[-2], mask =mask.flatten(), rmax=rmax )
        # tel.plot_ray_paths(ax, front_panel=tel.panels[0], rear_panel=tel.panels[-2], mask =mask.flatten(), rmax=rmax, c="grey", linewidth=0.5 )

        # Punto específico para centrar el gráfico
        center_x, center_y, center_z = x, y, z

        # Establecer límites para centrar el gráfico
        radius = 2000  # Radio para ajustar los límites del gráfico
        ax.set_xlim(center_x - radius, center_x + radius)
        ax.set_ylim(center_y - radius, center_y + radius)

    ax.set_aspect('equal')
    plt.show()

    