import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np 
import pandas as pd
import pickle
from scipy.interpolate import griddata
from scipy.io import loadmat
from tqdm import tqdm
#package modules
from acceptance.acceptance import geometrical_acceptance
from cli import get_common_args
from config import DATA_DIR,STRUCT_DIR
from telescope import DICT_TEL
from utils.common import Common
from utils.tools import print_file_datetime
from survey.run import RunTomo
from raylength.func import load_from_hdf5

params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':1,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': False,
         'figure.figsize': (8,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
plt.rcParams.update(params)


def load_dataframe(csvfile):
    df = pd.read_csv(csvfile)
    print_file_datetime(csvfile)
    df.set_index("event_id", inplace=True)
    return df

def get_run_duration(df, t0=0):
    grp_evt_id = df.groupby('event_id')
    time_ev = grp_evt_id['timestamp'].first().to_numpy()
    time_off = (time_ev - min(time_ev)) + t0 
    range_time_off = [np.nanmin(time_off),np.nanmax(time_off)]
    t_start, t_end = range_time_off
    run_duration = t_end - t_start
    return run_duration

def get_counts(tel, conf, df):
    mask = (df['config'] == conf.name) 
    x, y = df["dx_dz"][mask], df["dy_dz"][mask]
    u_edges, v_edges = conf.u_edges, conf.v_edges
    # u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
    bins=(u_edges, v_edges)
    h2d, binx, biny = np.histogram2d(x, y, bins=bins, range=conf.range_uv)
    return h2d   
        
def process_configuration_brut(h5file, tel, run, conf, df, duration):
    counts = get_counts(tel, conf, df)
    d = dict(counts=counts, duration=duration)
    save_in_hdf5(
                    h5file,
                    tel,
                    run,
                    conf,
                    **d,
                )
    return d

def process_brut(h5file, tel, run):
    csvfile =  run.dirs["reco"] / "df_track.csv.gz"
    assert csvfile.exists(), f"{csvfile} not found"
    df = load_dataframe(csvfile)
    duration = get_run_duration(df, )
    for _, conf in tel.configurations.items():
        process_configuration_brut(
            h5file,
            tel,
            run,
            conf,
            df, 
            duration,
        )

def compute_integral_flux(counts, acceptance, duration, eps=1e-12):
    return counts/(acceptance*duration + eps)

def process_configuration_tomo(h5file, tel, run, conf, acceptance, points, values, counts, duration, rays_length):
    flux = compute_integral_flux(counts, acceptance, duration) # 1 / [cm^2.sr.s]
    flux_min, flux_max = np.nanmin(points[:, 1]), np.nanmax(points[:, 1])
    flux = np.clip(flux, flux_min, flux_max)
    theta = tel.zenith_matrix[conf.name]
    xi = np.vstack((theta.ravel(), flux.ravel())).T
    opacity = griddata(points, values, xi=xi, method='linear') # mwe = hg/cm^2
    mean_density = np.zeros_like(opacity)
    rays_length = rays_length.reshape(conf.shape_uv).T.ravel()
    if rays_length is not None: 
        m = rays_length > 1e1
        mean_density[m] = opacity[m] / rays_length[m]
    else: rays_length = np.zeros_like(opacity)
    d = {"flux": flux, "opacity": opacity, "rays_length": rays_length, "mean_density": mean_density}
    save_in_hdf5(h5file,
                    tel,
                    run,
                    conf,
                    **d,)

def process_tomo(h5file, tel, run, acceptance, dirs, summed_counts, summed_durations, h5file_raylength):
    fluxop_file= dirs["structure"]["flux"] / "IntegralFluxVsOpAndZaStructure_Corsika.mat"
    fluxop_struct = loadmat(str(fluxop_file))['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'] [0][0]
    theta_grid_tomo = fluxop_struct[0] #zenith angle
    opacity_log10 = fluxop_struct[1] #opacity shape: (bins_ene, bins_op) = (100, 600)
    opacity_grid = np.exp( np.log(10) * opacity_log10 )
    flux_grid_tomo = fluxop_struct[2]
    #opacity interpolation from (theta, flux, opacity) grid
    X,Y, values = theta_grid_tomo.ravel(), flux_grid_tomo.ravel(), opacity_grid.ravel()
    points = np.vstack((X.ravel(), Y.ravel())).T
    for _, conf in tel.configurations.items():
        rays_length = load_from_hdf5(h5file_raylength, tel.name, conf.name)
        process_configuration_tomo(
                    h5file, tel, run, conf, 
                    acceptance[conf.name],
                    points, values,
                    summed_counts[conf.name],
                    summed_durations[conf.name], 
                    rays_length,
        )

def compute_experimental_acceptance(tel, conf, counts, duration, theta_grid_os, flux_grid_os):
    theta = tel.zenith_os_matrix[conf.name]
    flux = np.interp(theta, theta_grid_os, flux_grid_os)
    return counts/(flux*duration)

def process_calib(h5file, tel, run, dirs):
    dict_brut_calib = h5file[tel.name][run.name]
    fluxos_file = dirs["structure"]["flux"] / "openSkyFluxStructure.mat"
    fluxos_struct = loadmat(str(fluxos_file))["openSkyFluxStructure"][0][0]
    theta_grid_os = fluxos_struct[0].ravel() * np.pi/180
    flux_grid_os = fluxos_struct[1].ravel() #
    D = {}
    for _, conf in tel.configurations.items():
        counts, duration = dict_brut_calib[conf.name]["counts"], dict_brut_calib[conf.name]["duration"]
        acc = compute_experimental_acceptance(tel, conf, 
                                        counts, duration, 
                                        theta_grid_os, flux_grid_os)
        D[conf.name] = acc
        save_in_hdf5(h5file, tel, run, conf, **{"acceptance":acc})
    return D

def combine_tomo_data(h5file, tel, tomo_runs):
    summed_counts = {}
    summed_durations = {}
    for conf in tel.configurations.values():
        total_counts = None
        total_duration = 0.0
        for run in tomo_runs:
            d = load_values_from_hdf5(h5file, tel.name, run.name, conf.name, ["counts", "duration"])
            if total_counts is None:
                total_counts = d["counts"].copy()
            else:
                total_counts += d["counts"]
            total_duration += d["duration"][0]
        summed_counts[conf.name] = total_counts
        summed_durations[conf.name] = total_duration
    return summed_counts, summed_durations

def process_telescope(h5file, tel, dirs, raylength):
    tel.compute_angular_coordinates()
    dict_runs = survey.runs[tel.name]
    for _, run in dict_runs.items():
        if run.name == "calib" or run.name.startswith("tomo"):
            process_brut(h5file, tel, run)
    run_calib = dict_runs.get("calib")
    if not run_calib : 
        return
    acceptance = process_calib(h5file, tel, run_calib, dirs)
    ## Find all tomo runs
    tomo_runs = [run for run in dict_runs.values() if run.name.startswith("tomo")]
    if not tomo_runs:
        return
    summed_counts, summed_durations = combine_tomo_data(h5file, tel, tomo_runs)
    combined_run = RunTomo(name="tomo_combined", path=tomo_runs[0].path.parent) if len(tomo_runs) > 1 else tomo_runs[0]
    process_tomo(h5file, tel, combined_run, acceptance, dirs, summed_counts, summed_durations, raylength)
    
def save_in_hdf5(h5file, tel, run, conf, **kwargs):
    grp_tel = h5file.require_group(tel.name)
    grp_run = grp_tel.require_group(run.name)
    grp_conf = grp_run.require_group(conf.name)
    if kwargs is None: return 
    for k, v in kwargs.items():
        if not isinstance(v, np.ndarray): v=np.array([v])
        grp_conf.create_dataset(k, data=v)

def load_values_from_hdf5(file, tel_name, run_name, conf_name, keys:list=[]):
    grp = file[tel_name][run_name][conf_name]
    return load_all_datasets_from_group(grp, keys)

def load_all_datasets_from_group(grp, keys:list=[]):
    """
    Charge tous les datasets du groupe HDF5 spécifié.
    Si keys est vide, retourne un dict avec tous les datasets.
    Sinon, retourne un dict avec seulement les clés spécifiées.
    """
    if len(keys)==0:
        return {k: v[:] for k, v in grp.items() if isinstance(v, h5py.Dataset)}
    d = {}
    for k in keys:
        if k in grp and isinstance(grp[k], h5py.Dataset):
            d[k] = grp[k][:]
    return d

def set_norm(arr):
    vmin, vmax = np.nanmin(arr[arr!=0]), np.nanmax(arr)
    norm = LogNorm(vmin, vmax) if vmax/vmin > 2e1 else Normalize(vmin, vmax)
    return norm

def plot_image(ax, array, u, v):
    im = ax.pcolormesh(u,v, array, norm=set_norm(array))
    cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-2,loc = 'right')
    cb = plt.colorbar(im, cax=cax, extend='max')
    # cb.set_label(label)

def plot_configuration(axs, h5file, tel, run_name, str_image, mask=None):
    configurations=tel.configurations.items()
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    assert len(axs) == len(configurations), "Axes shape does not match number of configurations"
    for j,(_, conf) in enumerate(configurations):
        ax = axs[j]
        d = load_values_from_hdf5(h5file, tel.name, run_name, conf.name)
        assert str_image in d.keys(), f"{str_image} not in {d.keys()}"
        array = d[str_image]
        array = np.flipud(array) if (tel.name == "SXF") or (tel.name == "OM") else array
        u_edges, v_edges = conf.u_edges, conf.v_edges
        u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
        m = mask[conf.name].reshape(conf.shape_uv) if mask is not None else np.ones(conf.shape_uv, dtype=bool)
        array = array.reshape(conf.shape_uv)
        array[~m] = np.nan
        plot_image(ax, array, u,v)
        ax.label_outer()
        ax.set_aspect('equal')

# def set_mask_rays(voxray, tel):
#     voxray_tel = voxray[tel.name]
#     configurations=tel.configurations.items()
#     mask = {}
#     rays_length = {}
#     for j,(_, conf) in enumerate(configurations):
#         rl = np.array(voxray_tel[conf.name]["rays_length"])
#         mask[conf.name] = (1e1 < rl ) #& (rays_length <= 1e3)
#         rays_length[conf.name] = rl 
#     return mask, rays_length

if __name__ == "__main__":

    args = get_common_args(save=False)
    cmn = Common(args)
    survey = cmn.survey
    tel = cmn.telescope
    struct_dir = STRUCT_DIR / survey.name
    dirs = {
        "structure":{
                "voxel": struct_dir/"voxel",
                "tel": struct_dir/"telescope",
                "flux": struct_dir/"flux",
            },
       "data":  DATA_DIR,
        }   

    # dtel = {"SNJ": DICT_TEL["SNJ"]}
    runs = survey.runs[tel.name]

    mask_rays = None
    # h5file_voxray = dirs["structure"]["voxel"] / f"real_telescopes_voxel_ray_matrices_vox4m.h5"
    # with h5py.File(h5file_voxray) as fvr: 
    #     mask_rays, rays_length = set_mask_rays(fvr, tel)

    h5file_muo = dirs["data"] / tel.name / f"muography.h5"
    h5file_raylength = dirs["structure"]["tel"] / f"topo_roi_real_telescopes_rays_length.h5"
    print_file_datetime(h5file_raylength)
    with h5py.File(h5file_muo, "w") as file_muo:
        with h5py.File(h5file_raylength, "r") as file_raylength: 
            # for i, tel in tqdm(enumerate(dtel.values()), total=len(dtel), desc="Telescopes"):
            process_telescope(
                file_muo,
                tel,
                dirs, 
                file_raylength
                )
    print(f"Saved {h5file_muo}")

    ###Test read h5file
    ncols,nrows = len(tel.configurations),1
    
    run_calib=runs.get('calib')
    with h5py.File(h5file_muo, "r") as fmuo: 
        for str_image in ["counts","acceptance"]:
            fig, axs = plt.subplots(ncols=ncols,nrows=nrows, figsize=(6*ncols, 6*nrows), sharex=True, sharey=True)#constrained_layout=True)
            plot_configuration(axs, fmuo, tel, run_calib.name, str_image)
            fout_png = run_calib.dirs["png"] / str(str_image+".png")
            fig.savefig(fout_png)
            print(f"Saved {fout_png}")
            plt.close()


    run_tomo = runs.get('tomo')
    print_file_datetime(h5file_muo)
    with h5py.File(h5file_muo) as fmuo: 
        for str_image in ["counts","flux","opacity","rays_length","mean_density"]:
            fig, axs = plt.subplots(ncols=ncols,nrows=nrows, figsize=(6*ncols, 6*nrows), sharex=True, sharey=True)#constrained_layout=True)
            plot_configuration(axs, fmuo, tel, run_tomo.name, str_image, None)
            fout_png = run_tomo.dirs["png"] / str(str_image+".png")
            fig.savefig(fout_png)
            print(f"Saved {fout_png}")
            plt.close()
