import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np 
import pandas as pd
import pickle
from tqdm import tqdm
#package modules
from acceptance.acceptance import GeometricalAcceptance
from cli import get_common_args
from data import RawData
from flux import FluxModel
from utils.common import Common
from utils.tools import print_file_datetime

params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':2,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': False,
         'figure.figsize': (8,8),
         }
plt.rcParams.update(params)


if __name__ == "__main__":

    args = get_common_args(save=False)
    cmn = Common(args)
    survey = cmn.survey
    tel = cmn.telescope
    runs = survey.runs[tel.name]
    dict_conf = tel.configurations
    tel.compute_angle_matrix()
    file_pkl = cmn.pkl_path /  "images_corr.pkl"
    print_file_datetime(file_pkl)
    with open(file_pkl, 'rb') as f:
        dict_im = pickle.load(f)

    file_osflux = survey.flux[tel.name]
    print_file_datetime(file_osflux)
    with open(file_osflux, 'rb') as f:
        flux_os = pickle.load(f)
    print("flux_os", flux_os)


    # energy_bins = np.logspace(1e-2, 1000, 100)
    # fm = FluxModel(altitude=tel.altitude)
    # zenith_matrix = tel.zenith_matrix["3p1"]
    # arr_theta = zenith_matrix.ravel()
    # arr_flux = np.zeros_like(arr_theta)
    # for i, t in tqdm(enumerate(arr_theta), total=len(arr_theta), desc="Flux "):
    #     f_i = fm.ComputeOpenSkyFlux(t, model="guan")
    #     arr_flux[i] = f_i
    