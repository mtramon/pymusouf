#!/usr/bin/python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
# package module(s)
from config import STRUCT_DIR, DATA_DIR
from raylength.func import load_raylength_from_hdf5
from survey import CURRENT_SURVEY
from topography import build_interpolator_topography, interpolate_topography
from tv import InversionTV
from voxelgrid import load_voxel_grid
from utils.tools import print_file_datetime, check_array_order
from processing.func import test_run_key, set_norm
from processing.muography import load_values_from_hdf5

if __name__ == "__main__":
    
    survey = CURRENT_SURVEY
    dir_survey = STRUCT_DIR /  survey.name   
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"
    dir_tel = dir_survey / "telescope"
    dir_inv = dir_model / "inversion"
    dir_inv.mkdir(parents=True, exist_ok=True)
    dir_out = dir_inv / "real"
    dir_out.mkdir(parents=True, exist_ok=True)
    vs = int(sys.argv[1]) if len(sys.argv) >1 else 32  # voxel size in m (edge length)

    input_vtk = dir_voxel / f"topo_voi_vox{vs}m.vts"

    grid, geom = load_voxel_grid(input_vtk)
    mask_voi = geom.mask_voxel

    data_concat = None
    M_concat = None
    basename = f"real_telescopes"   

    dtel = survey.telescopes
    detectors_intercept = np.zeros_like(mask_voi) 
    det_coords = []

    h5file_voxray = dir_voxel / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"
    str_tel_type = basename.split("_")[0]

    h5file_raylength = dir_tel / f"topo_roi_{basename}_rays_length.h5"
    
    str_reg_type = "tv"
    file_out_npy = dir_out / f"model_vox{vs}m_{str_reg_type}_{str_tel_type}.npz"
    basename_out = file_out_npy.stem

    c = 0
    with h5py.File(h5file_voxray, "r") as fh5_voxray : 

        with h5py.File(h5file_raylength, "r") as fh5_raylength: 
        
            for tel_name, tel in tqdm(dtel.items(), desc="Data"):
        
                h5file_muo = DATA_DIR / tel.name / f"muography.h5"
                tel.compute_angular_coordinates()
                det_coords.append(tel.coordinates)
                x0, y0, z0 = tel.coordinates
                
                with h5py.File(h5file_muo, "r") as fh5_muo:

                    run_name = test_run_key(tel_name, fh5_muo)

                    for i, (conf_name, conf) in enumerate(tel.configurations.items()):
                        ze_matrix = tel.zenith_matrix[conf_name]
                        mask_rays = (np.rad2deg(ze_matrix) <= 90).ravel()
                        nu, nv = conf.shape_uv
                        u_edges, v_edges = conf.u_edges, conf.v_edges
                        u, v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
                        voxray = fh5_voxray[tel_name][conf_name]
                        weights = voxray["data"]
                        indices = voxray["indices"]
                        indptr = voxray["indptr"]
                        shape = voxray["shape"]
                        G_uv = np.array(voxray["acceptance"])
                        rays_length = load_raylength_from_hdf5(fh5_raylength, tel_name, conf_name)
                        # rays_length = np.array(voxray["rays_length"])
                        nvox, npix = shape
                        M = sp.csr_matrix(
                            (weights, indices, indptr),
                            shape=shape
                        ).T
                        M_csc = M.tocsc()
                        detectors_intercept += (np.diff(M_csc.indptr) > 0)
                        del M_csc
                        opacity = np.zeros(npix)
                        M_norm = M.multiply(1.0 / G_uv[:, None])
                        d = load_values_from_hdf5(fh5_muo, tel.name, run_name, conf.name)
                        opacity, unc = np.array(d["opacity"])[0] *1e3 # convert from mwe to kg/m^2
                        print(type(opacity), opacity.shape, "Opacity loaded from hdf5")
                        print(f"{tel_name} {conf_name} opacity min, max: ", np.nanmin(opacity[mask_rays]), np.nanmax(opacity[mask_rays]))
                        print(f"{tel_name} {conf_name} uncertainty min, max: ", np.nanmin(unc[mask_rays]), np.nanmax(unc[mask_rays]))
                        mask_rays_length = (1e1 < rays_length) & (rays_length < 1e3)
                        # print(np.all(mask_rays_length==False))
                        mask_rays = mask_rays & mask_rays_length
                        mean_opacity, std_opacity = np.mean(opacity[mask_rays]), np.std(opacity[mask_rays])
                        opacity[~mask_rays] = 0
                        unc[~mask_rays] = 0
                        # unc = np.zeros_like(opacity)
                        # unc[mask_rays] = np.clip(0.02*opacity[mask_rays], 1e-4*mean_opacity, 0.2*mean_opacity)
                        # opacity += unc
                        if c==0 : 
                            data_concat = opacity
                            unc_concat = unc
                            M_concat = M_norm   
                        else: 
                            data_concat = np.hstack((data_concat, opacity ))
                            unc_concat = np.hstack((unc_concat, unc))
                            M_concat = sp.vstack((M_concat, M_norm), format='csr')
                        c+=1
    print(c, "configurations loaded")
    x_edges, y_edges, z_edges = geom.x_edges, geom.y_edges, geom.z_edges
    nvx, nvy, nvz = len(x_edges)-1, len(y_edges)-1, len(z_edges)-1
    x =(x_edges[:-1] + x_edges[1:])/2
    y =(y_edges[:-1] + y_edges[1:])/2
    z =(z_edges[:-1] + z_edges[1:])/2
    cx =(x_edges[0] + x_edges[-1])/2
    cy =(y_edges[0] + y_edges[-1])/2
    cz =(z_edges[0] + z_edges[-1])/2
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)    
    ORDER = "F" #each voxel volume mask needs to be in Fortran order (to match M matrix building)
    mask_center = (R <= 500).reshape(-1,order=ORDER)
    zmax = np.max(z)
    mask_z = (Z > abs(zmax-300)).reshape(-1,order=ORDER)
    weight_concat = np.array(M_concat.sum(axis=0)).reshape(-1,order=ORDER)
    mask_weight = (weight_concat > 0)
    rays_intercept = np.diff(M_concat.tocsc().indptr)
    # mask_rays_intercept = (rays_intercept <= np.percentile(rays_intercept, 99.9)) 
    mask_det_intercept = (detectors_intercept >= 3)
    print("Data and model shapes: ", data_concat.shape, M_concat.shape)
    print("Data min, max: ", np.nanmin(data_concat), np.nanmax(data_concat))
    print("Unc min, max: ", np.nanmin(unc_concat), np.nanmax(unc_concat))
    opaque = np.where((data_concat > 0) & (unc_concat > 0))[0]
    ndata = len(opaque)
    print(f"Number of opaque voxels: {ndata}")
    data_concat = data_concat[opaque]
    unc_concat = unc_concat[opaque]
    print(f"Number of opaque voxels: {data_concat.shape}")
   
    mask_voxel = mask_voi & mask_center & mask_z & mask_det_intercept & mask_weight #& mask_rays_intercept

    coords = np.vstack((X.ravel(order=ORDER), Y.ravel(order=ORDER), Z.ravel(order=ORDER))).T

    sensitivity = np.zeros(nvox)
    diag_data = np.zeros(nvox)
    
    _active = np.where(mask_voxel)[0]
    _M = M_concat[opaque][:, _active]
    print("After filtering : nrays, nvoxels = ", _M.shape)

    # diag(M^T C_d^-1 M) = # diag(M^T W M)
    Cd_inv = 1 / (unc_concat**2)
    diag_data[_active] = np.array(_M.power(2).T @ Cd_inv)
    sensitivity[_active] = np.sqrt(diag_data[_active])    
    sthresh = np.percentile(sensitivity[_active], 1)
    mask_voxel = mask_voxel #& (sensitivity>sthresh)    
    active = np.where(mask_voxel)[0]    
    sens_active = sensitivity[active]

    M_concat = M_concat[opaque][:, active]
    nvox_active = len(active)

    #### PONDERATION DES VOXELS
    det_coords = np.asarray(det_coords)
    det_tree = cKDTree(det_coords)
    d, _ = det_tree.query(coords[active])
    dist_min = min(d)#1e2

    # Profondeur depuis la topographie
    # file_dem = dir_dem / "topo_roi.vts"
    # interp = build_interpolator_topography(file=file_dem)
    # Z_interp = interpolate_topography(interp, X, Y).ravel(ORDER)[active] #
    # z_coords = coords[active,2]
    # depth = np.maximum(0.0, Z_interp - z_coords)

    wd = ((d - dist_min) / (d.max() - dist_min + 1e-8))**2
    wd = np.clip(wd, 0, 1)
    weights = wd

    # Sensibilité (adoucie)
    s = sens_active.copy()
    s = s / np.max(s)
    beta=2
    ws = 1.0 / (s + 0.05)**beta
    # Normalisation + clipping (crucial)
    ws = ws / np.max(ws)
    ws = np.clip(ws, 0, 2)
    # Combinaison multiplicative
    gamma = 0.3
    # weights = wd * (1 + gamma * ws)

    # Longueur traversée
    length = np.array(M_concat.sum(axis=0)).ravel()
    w_len = 1 / (length + 1e-12)**0.5
    # Nombre de rayons
    nrays = rays_intercept[active]
    w_rays = 1 / (nrays + 1)**0.5
    # Combinaison
    # weights = wd * (1 + 0.3*w_len + 0.2*w_rays)
    # d0 = 10*vs
    # wd = 1 - np.exp(-((d-dist_min)/d0)**2)

    ####
    wz = np.zeros_like(s)

    s = sensitivity[_active]
    fig, ax = plt.subplots()
    x = sensitivity[sensitivity>0]
    vmin, vmax= np.min(x), np.max(x)
    print("sensitivity: \nmin, max:", vmin, vmax)
    logbins = np.logspace(start=np.log10(vmin), stop=np.log10(vmax), num=100)
    h,e= np.histogram(x, bins=logbins,)
    bc, w = (e[:-1]+e[1:])/2, abs(e[:-1]-e[1:])
    ax.bar(bc,  h, w )#label = f"$\\lambda_r$ = {l:.3e}\nmean={mean_post[i]:.3e}")
    ax.set_xscale("log")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Voxels")
    sthresh = np.percentile(sensitivity[_active], 10)
    sthresh = max(sthresh, 1e-3)
    ax.axvline(sthresh, color='red', linestyle='--', label=f"Sensitivity threshold = {sthresh:.3e}")
    fout_png = dir_out / f"{basename_out}_sensitivity.png"
    fig.savefig(fout_png)
    print(f"Saved {fout_png}")
    s = s / np.max(s)
    beta = 0.5
    mask_slow = (s < sthresh)
    # wz[mask_slow] = (1 - s[mask_slow])**beta
    s0 = sthresh
    gamma = 4
    wz = 1 / (1 + (s / s0)**gamma)

    print(np.count_nonzero(mask_slow), len(mask_slow))

    # normalisation (important)
    # wz = wz / (wz.max() + 1e-8)

    # fig,ax= plt.subplots()
    # idx_s = np.argsort(s)
    # ax.plot(s[idx_s],wz[idx_s])
    # ax.set_xscale("log")
    # # ax.set_yscale("log")
    # fig.savefig(dir_out/"weights_z.png", dpi=300)

    rho0 = 2000
    rho_min, rho_max = 100, 4000 #kg /m^3
    rho_range = [rho_min, rho_max]
    tv = InversionTV(
        M=M_concat,
        data=data_concat,
        unc=unc_concat,
        nx=nvx, ny=nvy, nz=nvz,
        mask=mask_voxel,
        weights=weights,
        weights_z = wz,
        rho0=np.ones(nvox_active) * rho0, 
        sensitivity=sens_active,
        density_range = rho_range
    )

    model_grid = vtk.vtkStructuredGrid()
    model_grid.ShallowCopy(grid)
    
    mask_active = np.zeros(nvox)
    mask_active[active] = 1
    active_vtk = numpy_support.numpy_to_vtk(
            mask_active.ravel(),
            deep=True,
            array_type=vtk.VTK_INT
        )
    active_vtk.SetName(f"active_voxels")
    model_grid.GetCellData().AddArray(active_vtk)

    # weight_vtk = numpy_support.numpy_to_vtk(
    #         weight_concat.astype(np.float32),
    #         deep=True,
    #         array_type=vtk.VTK_FLOAT

    #     )
    # weight_vtk.SetName(f"voxels_weight")
    # model_grid.GetCellData().AddArray(weight_vtk)

    rays_intercept_vtk = numpy_support.numpy_to_vtk(
            rays_intercept.astype(np.int32),
            deep=True,
            array_type=vtk.VTK_INT
        )
    rays_intercept_vtk.SetName(f"rays_nintercept")
    model_grid.GetCellData().AddArray(rays_intercept_vtk)
    
    sensitivity_vtk = numpy_support.numpy_to_vtk(
            sensitivity.astype(np.float64),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
    sensitivity_vtk.SetName(f"sensitivity")
    model_grid.GetCellData().AddArray(sensitivity_vtk)
    
    det_intercept_vtk = numpy_support.numpy_to_vtk(
            detectors_intercept.astype(np.int32),
            deep=True,
            array_type=vtk.VTK_INT

        )
    det_intercept_vtk.SetName(f"detectors_nintercept")
    model_grid.GetCellData().AddArray(det_intercept_vtk)

    det_distances = np.zeros(nvox)
    det_distances[active] = d
    det_distance_vtk = numpy_support.numpy_to_vtk(
            det_distances.astype(np.int32),
            deep=True,
            array_type=vtk.VTK_INT

        )
    det_distance_vtk.SetName(f"detectors_distance")
    model_grid.GetCellData().AddArray(det_distance_vtk)

    nlr, nmu = 10, 1
    lambda_reg = np.logspace(-4, -1, nlr)
    mu_damp = np.logspace(-2, -1, nmu)
    ntot = nmu * nlr
    models = np.zeros((ntot, nvox))
    regularizations = np.zeros((ntot, nvox))
    misfits_data, misfits_model = np.zeros(ntot), np.zeros(ntot)
    c = 0 
    # mu, mu_z = 0, 0.99
    mu_z = 0.5
    # rho_min, rho_max = min(rho_true_voi[active]), max(rho_true_voi[active])
    rho_init = np.ones(active.shape)*rho0
    for i, lr in tqdm(enumerate(lambda_reg), total=nlr, desc='Modeling'):  
        for j, mu in enumerate(mu_damp): 
            ix = i + j * nlr
            rho_post = tv.solve(rho_init=rho_init, lambda_reg=lr, mu=mu*lr, mu_z=mu_z*lr, max_iter=200 )
            misfits_data[ix], misfit_tv, misfit_damp = tv.misfit(rho_post, lambda_reg=lr, mu=mu*lr, mu_z=mu_z*lr)
            misfits_model[ix] = misfit_tv + misfit_damp
            models[ix, active] =  rho_post
            min_post, max_post = np.min(rho_post), np.max(rho_post)
            mean_post = np.mean(rho_post)
            print(f"{ix} ({lr:.2e}, {mu:.1e}): min, max post: {min_post}, {max_post}")
            print(f"{ix}, {lr:.2e}, {mu:.1e}): median=  {np.median(rho_post)}")
            model_vtk = models[ix].copy()
            model_vtk[active] = np.clip(rho_post, rho_min, rho_max)
            rho_vtk = numpy_support.numpy_to_vtk(
                model_vtk,
                deep=True,
                array_type=vtk.VTK_FLOAT, 
            )
            rho_vtk.SetName(f"{ix}_density_lr{lr:.2e}_mu{mu:.1e}")
            model_grid.GetCellData().AddArray(rho_vtk)
            c += 1
    model_grid.GetCellData().SetActiveScalars(
        f"density_lambda_{lambda_reg[0]:.3e}"
    )
    np.savez_compressed(file_out_npy,
                density_post = models,
                misfits=np.asarray((misfits_data, misfits_model)),
                ndata = ndata,
                lambda_reg = lambda_reg,
                mu_damp = mu_damp,
                sensitivity = sensitivity,
                mask_voxel = mask_active,
                rays_intercept=rays_intercept,
            )
    print(f"Saved npy {file_out_npy}")
    # '''
    # file_out_vts = dir_out / f"{basename_out}_{kernel[:4]}reg_l{int(length)}m_lambda_series.vts"
    file_out_vts = dir_out / f"{basename_out}.vts"
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(dir_out / file_out_vts))
    writer.SetInputData(model_grid)
    writer.Write()
    print(f"Saved structured grid {file_out_vts}")
    