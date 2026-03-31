import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import vtk
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from inversion.build_voxel_ray_matrix import generate_rays, filter_rays
from func import save_rays_length_hdf5, plot_configuration, load_from_hdf5

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

from scipy.interpolate import RegularGridInterpolator

from vtk.util.numpy_support import vtk_to_numpy

def extract_topo_from_2d_grid(grid):

    dims = [0, 0, 0]
    grid.GetDimensions(dims)
    nx, ny, nz = dims

    assert nz == 1, "La grille doit être 2D (nz=1)"

    pts = vtk_to_numpy(grid.GetPoints().GetData())
    pts = pts.reshape((nx, ny, 3), order="F")

    x = pts[:, 0, 0]
    y = pts[0, :, 1]
    z = pts[:, :, 2]  # topo

    return x, y, z

def build_topography_interpolator(grid):

    x, y, z = extract_topo_from_2d_grid(grid)

    interp = RegularGridInterpolator(
        (x, y),
        z,
        bounds_error=False,
        fill_value=-np.inf
    )

    return interp

def inside_topography(p, topo_interp):

    x, y, z = p
    z_topo = topo_interp((x, y))

    return z <= z_topo

def compute_ray_length_surface(origin, direction, topo_interp, max_dist, ds=2.0):

    t = 0.0
    inside_prev = inside_topography(origin, topo_interp)

    length = 0.0

    while t < max_dist:

        t_next = t + ds
        p_next = origin + t_next * direction

        inside_now = inside_topography(p_next, topo_interp)

        if inside_prev and inside_now:
            length += ds

        elif inside_prev != inside_now:
            # raffinement interface
            t0, t1 = t, t_next

            for _ in range(6):
                tm = 0.5 * (t0 + t1)
                pm = origin + tm * direction

                if inside_topography(pm, topo_interp) == inside_prev:
                    t0 = tm
                else:
                    t1 = tm

            if inside_prev:
                length += (t1 - t)

        inside_prev = inside_now
        t = t_next

    return length


def compute_all_rays_length(rays_dirs, origin, topo_interp, max_dist):

    nrays = rays_dirs.shape[0]
    rays_length = np.zeros(nrays)

    for i in tqdm(range(nrays), desc="Rays"):
        d = rays_dirs[i] / np.linalg.norm(rays_dirs[i])
        rays_length[i] = compute_ray_length_surface(
            origin=np.array(origin),
            direction=d,
            topo_interp=topo_interp,
            max_dist=max_dist
        )

    return rays_length


def process_configuration(
    tel,
    conf,
    topo_interp,
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
        topo_interp=topo_interp,
        max_dist=max_dist
    )
    return rays_length, subrays_is_main




def process_telescope(h5file, tel, grid, max_dist):
    tel.compute_angular_coordinates()

    # tel.adjust_height(
    #     geom.x_edges,
    #     geom.y_edges,
    #     geom.z_edges,
    #     geom.mask_voxel
    # )

    topo_interp = build_topography_interpolator(grid)

    for conf_name, conf in tel.configurations.items():
    
        rays_length, _ = process_configuration(
                    tel,
                    conf,
                    topo_interp,
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
    dtel = survey.telescope 
    input_vts = dir_dem / "topo_roi.vts"
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(input_vts))
    reader.Update()

    grid = reader.GetOutput()
    # input_vts = dir_voxel / "topo_voi_vox32m.vts"
    # grid, geom = load_voxel_grid(input_vts)
    # print(np.count_nonzero(geom.mask_voxel), len(geom.mask_voxel))
    # exit()
    # tel  = {"SNJ":survey.telescope ["SNJ"]}
    tel_name ="SB"
    tel = survey.telescope [tel_name]
    h5_path = dirs["voxel"] / f"{input_vts.stem}_{basename_tel}_rays_length.h5"
    with h5py.File(h5_path, "w") as fh5:
        # for i, tel in tqdm(enumerate(dtel.values()), total=len(dtel), desc="Telescopes"):
            process_telescope(
                fh5,
                tel,
                grid,
                max_dist=1200
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