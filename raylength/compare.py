import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
from matplotlib.colors import Normalize
import numpy as np
from tqdm import tqdm
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import check_array_order
from inversion.voxelgrid import load_voxel_grid
from inversion.build_voxel_ray_matrix import generate_rays, filter_rays
from func import save_rays_length_hdf5, plot_configuration, load_from_hdf5

if __name__ == "__main__":

    survey_name = CURRENT_SURVEY.name   
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"

    survey = CURRENT_SURVEY
    struct_dir = STRUCT_DIR / survey.name
    dirs = {
        "survey": struct_dir,
        "voxel": struct_dir/"voxel",
        "png": struct_dir / "png",
        "tel": struct_dir/ "telescope",
    }
    basename_tel = "real_telescopes"
    dtel = survey.telescope 
    input_vts = dir_dem / "topo_roi.vts"

    tel_name ="SB"
    tel = survey.telescope [tel_name]
    h5_path1 = dirs["voxel"] / f"{input_vts.stem}_{basename_tel}_rays_length.h5"
    h5_path2 = dirs["voxel"] / f"{input_vts.stem}_{basename_tel}_rays_length_fast.h5"
    ncols,nrows = len(tel.configurations),1
    fig, axs = plt.subplots(ncols=ncols,nrows=nrows, figsize=(6*ncols, 6*nrows), sharex=True, sharey=True)#constrained_layout=True)
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    with h5py.File(h5_path1) as fh5_1: 
        with h5py.File(h5_path2) as fh5_2: 
            for j,(_, conf) in enumerate(tel.configurations.items()):
                    ax = axs[j]        
                    delta = np.zeros(conf.npixels)
                    array1 = load_from_hdf5(fh5_1, tel.name, conf.name)  
                    array2 = load_from_hdf5(fh5_2, tel.name, conf.name)  
                    u_edges, v_edges = conf.u_edges, conf.v_edges
                    u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
                    m = array2 != 0 
                    delta[m] = (array1[m] - array2[m])/array2[m]  
                    im = ax.pcolormesh(u,v, delta.reshape(conf.shape_uv).T, norm=Normalize(vmin=-10, vmax=10))
                    cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-2,loc = 'right')
                    cb = plt.colorbar(im, cax=cax, extend='max')
                    ax.label_outer()
                    ax.set_aspect('equal')

    fig.savefig("ray_length_comparison.png")