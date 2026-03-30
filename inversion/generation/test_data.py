#!/usr/bin/python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import sys
from tqdm import tqdm
# package module(s)
from config import STRUCT_DIR
from inversion.model import load_density_model
from inversion.voxelgrid import load_voxel_grid
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
from synthetic_data import load_voxel_ray_matrix, compute_opacity, concat_results

if __name__ == "__main__":
    vs = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    survey_name = CURRENT_SURVEY.name
    dir_survey = STRUCT_DIR / survey_name

    dirs = {
        "survey": dir_survey,
        "dem": dir_survey / "dem",
        "voxel": dir_survey / "voxel",
        "model": dir_survey / "model",
        "tel": dir_survey / "telescope",
    }

    dirs["out"] = dirs["model"] / "test"
    dirs["out"].mkdir(parents=True, exist_ok=True)

    input_file = dirs["voxel"] / f"topo_center_anom_voi_vox{vs}m.vts"

    print_file_datetime(input_file)

    grid, geom = load_voxel_grid(input_file)

    nvox = len(geom.voxel_volume)
    rho0 = 1800 # in kg/m^3
    D0 = rho0 * np.ones(nvox, dtype=np.float64)  # density vector in kg/m^3
    mask_voi = geom.mask_voxel
    D0 = D0 * mask_voi

    density_model = load_density_model(input_file)

    basename = "real_telescopes"
    dtel = CURRENT_SURVEY.telescope

    # basename = f"toy_telescopes_s9506"   
    # fin_toytel = dir_tel / f"toy_telescopes_s9506_vox{vs}m.pkl"
    # with open(fin_toytel, 'rb') as f:
    #     dtel = pickle.load(f) 

    h5_path = dirs["voxel"] / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"

    data_concat = None
    unc_concat = None
    M_concat = None

    detectors_intercept = np.zeros_like(geom.mask_voxel)
    det_coords = []

    with h5py.File(h5_path) as fh5:

        for tel_name, tel in tqdm(dtel.items(), desc="Data"):

            tel.compute_angular_coordinates()
            det_coords.append(tel.coordinates)

            for conf_name, conf in tel.configurations.items():

                ze_matrix = tel.zenith_matrix[conf_name]
                mask_rays = (ze_matrix <= np.pi / 2).ravel()

                matrix = fh5[tel_name][conf_name]

                M, G_uv, rays_length = load_voxel_ray_matrix(matrix)

                M_csc = M.tocsc()
                detectors_intercept += (np.diff(M_csc.indptr) > 0)

                opacity, unc, M_norm = compute_opacity(
                    M,
                    G_uv,
                    density_model,
                    mask_rays
                )

                data_concat, unc_concat, M_concat = concat_results(
                    data_concat,
                    unc_concat,
                    M_concat,
                    opacity,
                    unc,
                    M_norm
                )