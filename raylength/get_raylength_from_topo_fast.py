import h5py
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from tqdm import tqdm
import vtk
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from inversion.build_voxel_ray_matrix import generate_rays
from func import save_rays_length_hdf5, plot_configuration
@njit
def interp_topo(x, y, x_grid, y_grid, z_grid):
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
        return -1e9  # extérieur
    ix = np.searchsorted(x_grid, x) - 1
    iy = np.searchsorted(y_grid, y) - 1
    ix = min(max(ix, 0), nx-2)
    iy = min(max(iy, 0), ny-2)
    x1, x2 = x_grid[ix], x_grid[ix+1]
    y1, y2 = y_grid[iy], y_grid[iy+1]
    z11 = z_grid[ix, iy]
    z12 = z_grid[ix, iy+1]
    z21 = z_grid[ix+1, iy]
    z22 = z_grid[ix+1, iy+1]

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    return (
        (1-tx)*(1-ty)*z11 +
        (1-tx)*ty*z12 +
        tx*(1-ty)*z21 +
        tx*ty*z22
    )

@njit
def compute_ray_length_fast(origin, direction, xg, yg, zg, max_dist, n_samples=128):

    dx, dy, dz = direction
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)

    dx /= norm
    dy /= norm
    dz /= norm

    # échantillonnage
    ts = np.linspace(0.0, max_dist, n_samples)

    vals = np.empty(n_samples)

    for i in range(n_samples):
        t = ts[i]
        x = origin[0] + t*dx
        y = origin[1] + t*dy
        z = origin[2] + t*dz

        topo = interp_topo(x, y, xg, yg, zg)
        vals[i] = z - topo

    length = 0.0

    for i in range(n_samples - 1):

        v0 = vals[i]
        v1 = vals[i+1]

        t0 = ts[i]
        t1 = ts[i+1]

        if v0 <= 0 and v1 <= 0:
            length += (t1 - t0)

        elif v0 * v1 < 0:
            # interpolation linéaire (rapide)
            alpha = abs(v0) / (abs(v0) + abs(v1))
            t_cross = t0 + alpha * (t1 - t0)

            if v0 <= 0:
                length += (t_cross - t0)
            else:
                length += (t1 - t_cross)

    return length

@njit
def compute_all_rays_fast(orig, rays_dirs, xg, yg, zg, max_dist):

    n = rays_dirs.shape[0]
    out = np.zeros(n)

    for i in range(n):
        out[i] = compute_ray_length_fast(
            orig,
            rays_dirs[i],
            xg,
            yg,
            zg,
            max_dist
        )
    
    return out

def prepare_topo_arrays(grid):

    from vtk.util.numpy_support import vtk_to_numpy

    dims = [0,0,0]
    grid.GetDimensions(dims)
    nx, ny, nz = dims

    pts = vtk_to_numpy(grid.GetPoints().GetData())
    pts = pts.reshape((nx, ny, 3), order="F")

    xg = pts[:,0,0]
    yg = pts[0,:,1]
    zg = pts[:,:,2]

    return xg, yg, zg

def process_configuration(
    tel,
    conf,
    grid,
    max_dist
):

    subrays_dirs, _, _, subrays_is_main, _ = \
        generate_rays(tel, conf, nsr=1)

    xg, yg, zg = prepare_topo_arrays(grid)

    rays_length = compute_all_rays_fast(
        np.array(tel.coordinates),
        subrays_dirs[subrays_is_main],
        xg, yg, zg,
        max_dist=max_dist
    )
    
    return rays_length, subrays_is_main

def process_telescope(h5file, tel, grid, max_dist):
    tel.compute_angular_coordinates()

    for conf_name, conf in tel.configurations.items():
    
        rays_length, _ = process_configuration(
                    tel,
                    conf,
                    grid,
                    max_dist=max_dist
                )
        
        save_rays_length_hdf5(
            h5file,
            tel.name,
            conf_name,
            rays_length
        )

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
    dtel = survey.telescopes 
    input_vts = dir_dem / "topo_roi.vts"
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(input_vts))
    reader.Update()

    grid = reader.GetOutput()
    h5_path = dirs["tel"] / f"{input_vts.stem}_{basename_tel}_rays_length.h5"

    with h5py.File(h5_path, "w") as fh5:
        for i, tel in tqdm(enumerate(dtel.values()), total=len(dtel), desc="Telescopes"):
            process_telescope(
                fh5,
                tel,
                grid,
                max_dist=1200
            )
    print(f"Saved {h5_path}")
    
    ###Test read h5file
    tel_name = "SNJ"
    tel = survey.telescopes [tel_name]
    tel = dtel[tel_name]
    ncols,nrows = len(tel.configurations),1
    with h5py.File(h5_path) as fh5: 
        for str_image in ["rays_length"]:
            fig, axs = plt.subplots(ncols=ncols,nrows=nrows, figsize=(6*ncols, 6*nrows), sharex=True, sharey=True)#constrained_layout=True)
            plot_configuration(axs, fh5, tel)
            fout_png = str(str_image+"_fast.png")
            fig.savefig(fout_png)
            print(f"Saved {fout_png}")
            plt.close()