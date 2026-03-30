#!/usr/bin/python3
# -*- coding: utf-8 -*-
import h5py
import glob
import numpy as np
from pathlib import Path
import pickle
import re
import scipy.sparse as sp
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm
import vtk 
from vtk.util import numpy_support

# package module(s)
from config import STRUCT_DIR
from inversion.model import load_density_model
from inversion.voxelgrid import load_voxel_grid, get_coordinates
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime, check_array_order
from inversion.tv import InversionTV

def load_voxel_ray_matrix(matrix):
    weights = matrix["data"]
    indices = matrix["indices"]
    indptr = matrix["indptr"]
    shape = matrix["shape"]
    G_uv = np.array(matrix["acceptance"])
    rays_length = np.array(matrix["rays_length"])
    M = sp.csr_matrix(
        (weights, indices, indptr),
        shape=shape
    ).T
    return M, G_uv, rays_length

def compute_opacity(M, G_uv, density_model, mask_rays=None, noise_level:float=0.02):
    nrays = M.shape[0]
    if mask_rays is None: mask_rays = np.ones(nrays, dtype=bool)
    opacity = np.zeros(nrays)
    M_norm = M.multiply(1.0 / (G_uv[:, None]+1e-12))
    opacity[mask_rays] = M_norm.dot(density_model)[mask_rays]
    mask_rays = mask_rays & (opacity > 1e3)
    mean_opacity = np.nanmean(opacity[mask_rays])
    unc = np.zeros_like(opacity, dtype=np.float32)
    unc_min, unc_max =   1e-4 * mean_opacity, 0.2 * mean_opacity
    unc[mask_rays] = np.clip(
        noise_level * opacity[mask_rays],
        unc_min,
        unc_max
    )
    opacity += unc
    opacity[~mask_rays] = 0
    return opacity, unc, M_norm

def concat_results(data_concat, unc_concat, M_concat, opacity, unc, M_norm):
    if data_concat is None:
        data_concat = opacity
        unc_concat = unc
        M_concat = M_norm
    else:
        data_concat = np.hstack((data_concat, opacity))
        unc_concat = np.hstack((unc_concat, unc))
        M_concat = sp.vstack((M_concat, M_norm), format="csr")

    return data_concat, unc_concat, M_concat

def define_mask_voxel(coords, M, detectors_intercept=None, rays_intercept=None, order="F"):
    '''
    N.B. Each volume mask needs to be in Fortran "F" order to match M matrix ordering
    '''
    X,Y,Z = coords.XYZ.T
    cx,cy,cz = coords.centre.T
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)    
    mask_center = (R <= 500).reshape(-1,order=order)
    zmax = np.max(Z)
    mask_z = (Z > abs(zmax-300)).reshape(-1,order=order)
    weight_concat = np.array(M.sum(axis=0)).reshape(-1,order=order)
    mask_weight = (weight_concat > 0).reshape(-1,order=order)
    mask_det_intercept = np.ones(X.shape, dtype=bool, order=order)
    if detectors_intercept is not None:
        mask_det_intercept = (detectors_intercept >= 3)
    # rays_intercept = np.diff(M.tocsc().indptr)
    # mask_rays_intercept = np.ones(X.shape, dtype=bool, order=order)
    # if rays_intercept is not None:
    #     mask_rays_intercept = (rays_intercept <= np.percentile(rays_intercept, 99.9)) 
    mask_voxel = mask_center & mask_z & mask_weight & mask_det_intercept  #& mask_rays_intercept
    return mask_voxel

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
    dirs["dataset"] = dirs["model"] / "synthetic" / "dataset"
    dirs["dataset"].mkdir(parents=True, exist_ok=True)
    dirs["train"] = dirs["model"] / "synthetic" / "training"
    dirs["train"].mkdir(parents=True, exist_ok=True)



    # input_vtk = dirs["model"] / f"ElecCond_topo_voi_vox{vs}m.vts"
    # input_vtk = dirs["voxel"] / f"topo_center_anom_voi_vox{vs}m.vts"
    input_vtk = dirs["voxel"] / f"topo_voi_vox{vs}m.vts"
    
    print_file_datetime(input_vtk)

    grid, geom = load_voxel_grid(input_vtk)
    mask_voi = geom.mask_voxel

    basename = "real_telescopes"
    dtel = CURRENT_SURVEY.telescope

    basename = f"toy_telescopes_s9506"   
    fin_toytel = dirs["tel"] / f"toy_telescopes_s9506_vox{vs}m.pkl"
    with open(fin_toytel, 'rb') as f:
        dtel = pickle.load(f) 

    voxel_coords = get_coordinates(geom)
    det_coords = [tel.coordinates for _, tel in dtel.items()] 

    ####MODELS  
    list_model_files = glob.glob(str(dirs["dataset"] / "model_*.vts"))
    def get_index_model(filename): 
        pattern = re.compile(r"model_(\d{3})\.vts")
        match = re.search(pattern, filename) 
        if match: return int(match.group(1)) 
        return -1 # Liste triée des fichiers VTI 
    list_model_files = [f for f in sorted(list_model_files, key=get_index_model) if get_index_model(f)>-1]

    list_model_files.extend([dirs["voxel"] / f"topo_center_anom_voi_vox{vs}m.vts", dirs["model"] / f"ElecCond_topo_voi_vox{vs}m.vts"])

    model_pairs = []

    for i, model_file in tqdm(enumerate(list_model_files), total=len(list_model_files), desc="Models") :
        # if i > 3: continue
        model_file = Path(model_file)
        basename_model = model_file.stem
        if not basename_model.startswith("model"): 
            continue
        model_truth = load_density_model(model_file) 

        ####DATA GENERATION
        h5_path = dirs["voxel"] / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"
        str_tel_type = basename.split("_")[0]
        data_concat = None
        unc_concat = None
        M_concat = None
        detectors_intercept = np.zeros_like(geom.mask_voxel)

        with h5py.File(h5_path) as fh5:
            # for tel_name, tel in tqdm(dtel.items(), desc="Data"):
            for tel_name, tel in dtel.items():
                for conf_name, conf in tel.configurations.items():
                    matrix = fh5[tel_name][conf_name]
                    M, G_uv, rays_length = load_voxel_ray_matrix(matrix)
                    M_csc = M.tocsc()
                    detectors_intercept += (np.diff(M_csc.indptr) > 0)
                    opacity, unc, M_norm = compute_opacity(
                        M,
                        G_uv,
                        model_truth * mask_voi,
                        None
                    )
                    data_concat, unc_concat, M_concat = concat_results(
                        data_concat,
                        unc_concat,
                        M_concat,
                        opacity,
                        unc,
                        M_norm
                    )


        mask_voxel = define_mask_voxel(voxel_coords, M_concat, detectors_intercept)
        mask_voxel = mask_voi & mask_voxel
        active = np.where(mask_voxel)[0]    
        opaque = np.where(data_concat > 0)[0]
        nrays, nvox = M_concat.shape
        data_concat = data_concat[opaque]
        # print(f"min, max data_concat = {np.min(data_concat):.3e}, {np.max(data_concat):.3e}")
        unc_concat = unc_concat[opaque]
        ####
        ####DATA INVERSION
        _active = np.where(mask_voxel)[0]
        sensitivity = np.zeros(nvox)
        diag_data = np.zeros(nvox)
        Cd_inv = 1 / (unc_concat**2)
        _M = M_concat[opaque][:, _active]
        diag_data[_active] = np.array(_M.power(2).T @ Cd_inv)
        sensitivity[_active] = np.sqrt(diag_data[_active])    
        sthresh = np.percentile(sensitivity[_active], 0.1)
        # mask_voxel = mask_voxel & (sensitivity>sthresh)    
        active = np.where(mask_voxel)[0]    
        sens_active = sensitivity[active]
        M_concat = M_concat[opaque][:, active]

        det_coords = np.asarray(det_coords)
        det_tree = cKDTree(det_coords)
        d, _ = det_tree.query(voxel_coords.XYZ[active])
        dist_min = min(d)
        d0 = 10*vs
        wd = 1 - np.exp(-((d-dist_min)/d0)**2)
        weights = wd

        nvox_active = np.count_nonzero(mask_voxel)
        # print(check_array_order(model_truth), check_array_order(mask_voi), check_array_order(mask_voxel))
        rho_median = np.median(model_truth[active])
        rho_mean = np.mean(model_truth[active])
        # print(rho_median, rho_mean)
        nx, ny, nz = len(geom.x_edges), len(geom.y_edges), len(geom.z_edges)
        nvx, nvy, nvz = nx-1, ny-1, nz-1
        nvox = nvx* nvy* nvz
        D0 = np.ones(nvox_active) * rho_median
        tv = InversionTV(
            M=M_concat,
            data=data_concat,
            unc=unc_concat,
            nx=nvx, ny=nvy, nz=nvz,
            mask=mask_voxel,
            weights=weights,
            rho0=D0
        ) 
        nlr, nmu = 1, 1
        lambda_reg = np.logspace(-4, -1, nlr)
        mu_damp = np.logspace(-4, -1, nmu)
        npost = nmu * nlr
        models_res = np.zeros((npost, nvox))
        misfits_data, misfits_model = np.zeros(npost), np.zeros(npost)
        c = 0 
        mu, mu_d = 0, 0 
        rho_min, rho_max = min(model_truth[active]), max(model_truth[active])
        Di = np.ones(nvox_active) * rho_median
        # for i, lr in tqdm(enumerate(lambda_reg), total=nlr, desc='Modeling'):
        for i, lr in enumerate(lambda_reg):
            for j, mu in enumerate(mu_damp): 
                ix = i + j*nlr
                rho_post = tv.solve(rho_init=Di, lambda_reg=lr, mu=mu*lr, max_iter=200 )
                models_res[ix, active] = rho_post
                misfits_data[ix], misfit_tv, misfit_damp = tv.misfit(rho_post, lambda_reg=lr, mu=mu*lr)
                misfits_model[ix] = misfit_tv + misfit_damp
                min_post, max_post = np.min(rho_post), np.max(rho_post)
                mean_post = np.mean(rho_post)
                # print(f"{ix} ({lr:.2e}, {mu:.1e}): min, max post: {min_post}, {max_post}")
                # print(f"{ix}, {lr:.2e}, {mu:.1e}): median=  {np.median(rho_post)}")
                c += 1
                
        ndata = len(opaque)
        ix_opt = np.argmin(abs(misfits_data/ndata-1))

        model_opt = models_res[ix_opt]
        min_post, max_post = np.min(model_opt[active]), np.max(model_opt[active])
        # print(f"{ix_opt} ({lr:.2e}, {mu:.1e}): min, max post_opt: {min_post}, {max_post}")
        # print(f"{ix_opt} ({lr:.2e}, {mu:.1e}): median post_opt:  {np.median(model_opt[active])}")
        
        file_out_npz = dirs["dataset"] / f"{basename_model}_reg.npz"
        np.savez(file_out_npz, density_truth=model_truth, density_reg=model_opt, shape=(nvx, nvy, nvz))

        if basename_model.startswith("model"):
            model_pairs.append((model_opt, model_truth)) 

        truth_npy = np.zeros(nvox)
        truth_npy[active] = model_truth[active]
        truth_vtk = numpy_support.numpy_to_vtk(
                truth_npy,
                deep=True,
                array_type=vtk.VTK_FLOAT, 
            )
        truth_vtk.SetName(f"density_truth")

        reg_npy = np.zeros(nvox)
        reg_npy[active] = np.clip(model_opt[active], rho_min, rho_max)
        reg_vtk = numpy_support.numpy_to_vtk(
            reg_npy,
            deep=True,
            array_type=vtk.VTK_FLOAT, 
        )
        reg_vtk.SetName(f"density_reg")

        model_grid = vtk.vtkStructuredGrid()
        model_grid.ShallowCopy(grid)
        model_grid.GetCellData().AddArray(truth_vtk)
        model_grid.GetCellData().AddArray(reg_vtk)

        basename_model = model_file.stem
        file_out_vts = dirs["dataset"] / f"{basename_model}_reg.vts"
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(str(file_out_vts))
        writer.SetInputData(model_grid)
        writer.Write()
        # print(f"Saved structured grid {file_out_vts}")

    if len(model_pairs) > 0 :
        result = np.array([pair[0] for pair in model_pairs]) # shape (N, nvox)
        truth = np.array([pair[1] for pair in model_pairs]) 
        fout_npz = dirs["train"] / "training_data.npz"
        np.savez(fout_npz, X=result, y=truth, mask=mask_voxel, active=active, shape=(nvx, nvy, nvz))
        print(f"Saved npz {fout_npz}")

    
