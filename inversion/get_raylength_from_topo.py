import h5py
from mpl_toolkits.axes_grid1 import inset_locator
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from pathlib import Path
from tqdm import tqdm
import vtk
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime, check_array_order
from voxelgrid import load_voxel_grid, load_dem_grid
from build_voxel_ray_matrix import generate_rays, filter_rays

def make_inside_function_voi(geom):
    mask = geom.mask_voxel
    nx = len(geom.x_edges) - 1
    ny = len(geom.y_edges) - 1
    nz = len(geom.z_edges) - 1

    def inside(x, y, z):
        ix = np.searchsorted(geom.x_edges, x) - 1
        iy = np.searchsorted(geom.y_edges, y) - 1
        iz = np.searchsorted(geom.z_edges, z) - 1
        if (ix < 0 or iy < 0 or iz < 0 or
            ix >= nx or iy >= ny or iz >= nz):
            return False
        idx = ix + nx * (iy + ny * iz)

        return mask[idx] > 0

    return inside

def compute_ray_length(origin, direction, inside_fn, max_dist, ds=2.0):
    t = 0.0
    inside_prev = inside_fn(*origin)
    length = 0.0
    while t < max_dist:
        t_next = t + ds
        p = origin + t_next * direction
        inside_now = inside_fn(*p)
        if inside_prev and inside_now:
            length += ds
        elif inside_prev != inside_now:
            # raffinement binaire pour trouver l’interface
            t0, t1 = t, t_next
            for _ in range(5):
                tm = 0.5 * (t0 + t1)
                pm = origin + tm * direction
                if inside_fn(*pm) == inside_prev:
                    t0 = tm
                else:
                    t1 = tm
            if inside_prev:
                length += (t1 - t)
        inside_prev = inside_now
        t = t_next
    return length

def compute_all_rays_length(rays_dirs, origin, inside_fn, max_dist):

    nrays = rays_dirs.shape[0]
    rays_length = np.zeros(nrays)

    for i in tqdm(range(nrays), desc="Rays"):
        d = rays_dirs[i] / np.linalg.norm(rays_dirs[i])
        rays_length[i] = compute_ray_length(
            origin=np.array(origin),
            direction=d,
            inside_fn=inside_fn,
            max_dist=max_dist
        )

    return rays_length


def process_configuration(
    tel,
    conf_name,
    conf,
    geom,
    inside_fn,
    max_dist
):

    subrays_dirs, _, _, subrays_is_main, _ = \
        generate_rays(tel, conf, nsr=1)

    # subrays_dirs, _, _, subrays_is_main = \
    #     filter_rays(
    #         subrays_dirs,
    #         np.ones(len(subrays_dirs)),
    #         np.arange(len(subrays_dirs)),
    #         subrays_is_main
    # )

    rays_length = compute_all_rays_length(
        subrays_dirs,
        origin=np.array(tel.coordinates),
        inside_fn=inside_fn,
        max_dist=max_dist
    )

    return rays_length, subrays_is_main

def save_rays_length_hdf5(h5file, tel_name, conf_name, rays_length):

    grp_tel = h5file.require_group(tel_name)
    grp = grp_tel.require_group(conf_name)

    if "rays_length" in grp:
        del grp["rays_length"]

    grp.create_dataset(
        "rays_length",
        data=rays_length,
        compression="gzip"
    )


def process_telescope(h5file, tel, geom, dirs, vtkfile):
    tel.compute_angular_coordinates()

    # tel.adjust_height(
    #     geom.x_edges,
    #     geom.y_edges,
    #     geom.z_edges,
    #     geom.mask_voxel
    # )

    inside_fn = make_inside_function_voi(geom)

    for conf_name, conf in tel.configurations.items():
    
        rays_length, mask_main = process_configuration(
                    tel,
                    conf_name,
                    conf,
                    geom,
                    inside_fn,
                    max_dist=1000.0
                )
        print(check_array_order(rays_length))
        save_rays_length_hdf5(
            h5file,
            tel.name,
            conf_name,
            rays_length
        )

def load_from_hdf5(h5file, tel_name, conf_name, key="rays_length"):
    grp = h5file[tel_name][conf_name]
    return np.array(grp[key])

def set_norm(arr):
    vmin, vmax = np.nanmin(arr[arr!=0]), np.nanmax(arr)
    norm = LogNorm(vmin, vmax) if vmax/vmin > 2e1 else Normalize(vmin, vmax)
    return norm

def plot_configuration(axs, h5file, tel, mask=None):
    configurations=tel.configurations.items()
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    assert len(axs) == len(configurations), "Axes shape does not match number of configurations"
    for j,(_, conf) in enumerate(configurations):
        ax = axs[j]
        array = load_from_hdf5(h5file, tel.name, conf.name)
        array = np.flipud(array) if (tel.name == "SXF") or (tel.name == "OM") else array.reshape(-1, order="C")
        u_edges, v_edges = conf.u_edges, conf.v_edges
        u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
        # m = mask[conf.name].reshape(conf.shape_uv) if mask is not None else np.ones(conf.shape_uv, dtype=bool)
        array = array.reshape(conf.shape_uv)
        # array[~m] = np.nan
        im = ax.pcolormesh(u,v, array)#norm=set_norm(array))
        cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-2,loc = 'right')
        cb = plt.colorbar(im, cax=cax, extend='max')
        ax.label_outer()
        ax.set_aspect('equal')

if __name__ == "__main__":

    dir_dem = Path("/Users/raphael/pymusouf/struct_link/soufriere/dem")
    input_vts = dir_dem / "topo_voi.vts"
    
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
    grid, geom = load_dem_grid(input_vts)
    # tel  = {"SNJ":survey.telescope ["SNJ"]}
    tel_name ="SB"
    tel = survey.telescope [tel_name]
    h5_path = dirs["voxel"] / f"{input_vts.stem}_{basename_tel}_rays_length.h5"
    with h5py.File(h5_path, "w") as fh5:
        # for i, tel in tqdm(enumerate(dtel.values()), total=len(dtel), desc="Telescopes"):
            process_telescope(
                fh5,
                tel,
                geom,
                dirs, 
                None,
            )
    print(f"Saved {h5_path}")


    ###Test read h5file
    tel = dtel[tel_name]
    ncols,nrows = len(tel.configurations),1
    with h5py.File(h5_path) as fh5: 
        for str_image in ["rays_length"]:
            fig, axs = plt.subplots(ncols=ncols,nrows=nrows, figsize=(6*ncols, 6*nrows), sharex=True, sharey=True)#constrained_layout=True)
            plot_configuration(axs, fh5, tel)
            fout_png = str(str_image+".png")
            fig.savefig(fout_png)
            print(f"Saved {fout_png}")
            plt.close()