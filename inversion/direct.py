#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Script for direct inversion, without solver (e.g. CG)
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import sys
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
import pickle
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from telescope import DICT_TEL
from utils.tools import print_file_datetime
titlesize=24
fontsize=24
params = {'legend.fontsize': fontsize,
          'legend.title_fontsize' : titlesize,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':titlesize,
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':1,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': True,
         'grid.alpha':0.3,
         'figure.figsize': (8,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
plt.rcParams.update(params)



if __name__ == "__main__":
    
    

    survey_name = CURRENT_SURVEY.name   
    print(f"Processing survey: {survey_name}")
    print(f"Survey structure directory: {STRUCT_DIR / survey_name}")
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"
    dir_tel = dir_survey / "telescope"
    dir_inv = dir_model / "inversion"
    dir_inv.mkdir(parents=True, exist_ok=True)
    vs = int(sys.argv[1]) if len(sys.argv) >1 else 32  # voxel size in m (edge length)
    # input_file = dir_voxel / f"topo_voi_vox{vs}m.vts"
    # input_file = dir_voxel / f"topo_bulge_voi_vox{vs}m.vts"
    # input_file = dir_model / f"ElecCond_CentralCube_aligned_voi_vox{vs}m.vts"
    # input_file = dir_model / f"ElecCond_topo_voi_vox{vs}m.vts"
    input_file = dir_voxel / f"topo_center_anom_voi_vox{vs}m.vts"

    print_file_datetime(input_file)
    
    # Read source structured grid
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(input_file))
    reader.Update()
    src_grid = reader.GetOutput()
    
    dx, dy, dz = np.ones(3)*vs
    xmin, xmax, ymin, ymax, zmin, zmax = src_grid.GetExtent()

    nx, ny, nz = xmax-xmin+1, ymax-ymin+1, zmax-zmin+1  
    pts = src_grid.GetPoints()
    xmin, xmax, ymin, ymax, zmin, zmax=pts.GetBounds()
    

    nedges = nx * ny * nz
    nvox = src_grid.GetNumberOfCells()
    nvx, nvy, nvz = nx-1, ny-1, nz-1
   
    # voxel_volume_vtk = src_grid.GetCellData().GetArray("voxel_volume")
    # voxel_volume = np.array(voxel_volume_vtk)
    voxel_density_vtk = src_grid.GetCellData().GetArray("density")
    voxel_density = np.array(voxel_density_vtk)
    
    mask_voi = voxel_density > 0   # True = voxel du volcan
    # mask_voi = mask_voi.astype(np.uint8)  # Numba-friendly

    D_1 = voxel_density * mask_voi  
    if D_1.max() < 1e3: 
        D_1 *= 1e3 # if density is in g/cm^3, convert to kg/m^3

    basename = f"real_telescopes"   
    dtel = CURRENT_SURVEY.telescopes

    # basename = f"toy_telescopes_s9506"   
    # fin_toytel = dir_tel / f"toy_telescopes_s9506_vox{vs}m.pkl"
    # with open(fin_toytel, 'rb') as f:
    #     dtel = pickle.load(f) 

    h5_path = dir_voxel / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"

    str_tel_type = basename.split("_")[0]

    r, c = 0, 0
    data_concat = None
    M_concat = None
    detectors_intercept = np.zeros(nvox, dtype=np.int32) 
    # det_coords = [] 
    with h5py.File(h5_path) as fh5:

        for tel_name, tel in tqdm(dtel.items(), desc="Data"):
            tel.compute_angular_coordinates()
            # det_coords.append(tel.coordinates)
            # if tel.name == "SXF": continue
            # if tel.name == "SBR": continue
            x0, y0, z0 = tel.coordinates
            for conf_name, conf in tel.configurations.items():
                # if r >= 5:continue
                ze_matrix = tel.zenith_matrix[conf_name].ravel()
                mask_rays = ze_matrix <= 90 * np.pi/180
                nu, nv = conf.shape_uv
                # if (conf_name == "4p") : continue
                u_edges, v_edges = conf.u_edges, conf.v_edges
                u, v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
                
                # voxray_file = dir_voxel / f"voxel_ray_matrix_{tel.name}_{conf_name}_vox{vs}m.npz"
                # voxray = np.load(voxray_file, allow_pickle=True) 
                voxray = fh5[tel_name][conf_name]

                weights = voxray["data"]
                indices = voxray["indices"]
                indptr = voxray["indptr"]
                shape = voxray["shape"]
                G_uv = np.array(voxray["acceptance"])
                nvox, npix = shape
                M = sp.csr_matrix(
                    (weights, indices, indptr),
                    shape=shape
                ).T #F-ordered
                M_csc = M.tocsc()
                detectors_intercept += (np.diff(M_csc.indptr) > 0)
                del M_csc
                opacity = np.zeros(npix)
                M_norm = M.multiply(1.0 / G_uv[:, None])
                # print(f"{tel_name}, {conf_name}: min, max M_norm.sum( axis=1)) = {np.min(M_norm.sum( axis=1)):.3e}, {np.max(M_norm.sum( axis=1)):.3e}")
                # M_norm = M
                # ix_rays = np.where(mask_rays)[0]
                # opacity[mask_rays] = M_norm.dot(D_1) [mask_rays]
                opacity = M_norm.dot(D_1) 

                # print(np.mean(opacity[mask_rays]))
                mask_rays = mask_rays & (opacity > 1e3)
                # print(np.min(M_norm.data), np.max(M_norm.data), M_norm.shape)
                # tau_model = M_norm @ D_0
                # print(f"tau_model mean {np.mean(tau_model):.3e}")
                # print(f"data mean {np.mean(opacity[mask_rays]):.3e}")
                mean_opacity, std_opacity = np.mean(opacity), np.std(opacity)
                # op_scaled = opacity[mask_rays]
                # op_scaled = (op_scaled-np.min(op_scaled)) / (np.max(op_scaled)-np.min(op_scaled))
                unc = np.zeros_like(opacity)
                # unc_rand = np.random.normal(0.05*opacity[mask_rays], 0.01*opacity[mask_rays])
                # unc[mask_rays] = np.clip(unc_rand, 1e-4*mean_opacity, 0.2*mean_opacity)
                # unc_model = 0.025 + 0.07/(1+np.exp(40*(op_scaled-0.02))) + 0.05/(1+np.exp(8*(0.9-op_scaled))) #noise model Lelièvre 2019
                # unc[mask_rays] = np.clip(unc_model*opacity[mask_rays], 1e-4*mean_opacity, 0.2*mean_opacity)
                # unc[mask_rays] = np.clip(unc_rand, 1e-4*mean_opacity, 0.2*mean_opacity)
                unc[mask_rays] = np.clip(0.01*opacity[mask_rays], 1e-4*mean_opacity, 0.2*mean_opacity)
                opacity += unc
                opacity[~mask_rays] = 0 
                if r==0 : 
                    data_concat = opacity
                    unc_concat = unc
                    M_concat = M_norm   
                else: 
                    data_concat = np.hstack((data_concat, opacity ))
                    unc_concat = np.hstack((unc_concat, unc))
                    M_concat = sp.vstack((M_concat, M_norm), format='csr')
                r+=1 
    print("All : nrays, nvoxels = ", M_concat.shape)
    ix = np.arange(nvx)
    iy = np.arange(nvy)
    iz = np.arange(nvz)
    x = xmin + (ix + 0.5) * vs
    y = ymin + (iy + 0.5) * vs
    z = zmin + (iz + 0.5) * vs
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    xc = xmin + nvx * vs / 2
    yc = ymin + nvy * vs / 2
    zc = zmin + nvz * vs / 2
    R = np.sqrt((X - xc)**2 + (Y - yc)**2)  
    ORDER = "F" #each voxel volume mask needs to be in Fortran order (to match M matrix building)
    mask_center = (R <= 500).reshape(-1,order=ORDER)
    mask_z = (Z > abs(zmax-300)).reshape(-1,order=ORDER)
    weight_concat = np.array(M_concat.sum(axis=0)).reshape(-1,order=ORDER)
    mask_weight = (weight_concat > 0)
    rays_intercept = np.diff(M_concat.tocsc().indptr)
    mask_rays_intercept = (rays_intercept <= np.percentile(rays_intercept, 99.9))
    mask_det_intercept = (detectors_intercept >= 1) 

    opaque = np.where(data_concat > 0 )[0]
    data_obs = data_concat[opaque]
    unc_obs = unc_concat[opaque]
    
    mask_voxel = mask_voi & mask_center & mask_z & mask_det_intercept & mask_weight & mask_rays_intercept
    coords = np.vstack((X.ravel(ORDER), Y.ravel(ORDER), Z.ravel(ORDER))).T

    Cd_inv = 1.0 / (unc_obs**2)
    # active = np.where(mask_voxel)[0]
    print("Full M: nrays, nvoxels = ", M_concat.shape)
    # A_sub = M_concat[opaque][:, active]
    diag_data = np.array(M_concat[opaque].power(2).T @ Cd_inv)
    # sensitivity = np.zeros(nvox)
    sensitivity = np.sqrt(diag_data)
    sthresh = np.percentile(sensitivity[mask_voxel], 5)
    print(f"Sensitivity threshold : {sthresh:.3e}")
    mask_voxel = mask_voxel & (sensitivity>sthresh)   
    active = np.where(mask_voxel)[0]
    
    A_sub = M_concat[opaque][:, active]
    print("Filtered sensitivity: nrays, nvoxels = ", A_sub.shape)
    
    A_sub= A_sub.toarray()
    coords_active = coords[active]
    
    rho_0 = 1800 # in kg/m^3
    D_0 = rho_0 * np.ones(nvox, dtype=np.float64)  # density vector in kg/m^3
    D_0 = D_0 * mask_voi
    rho0_sub = D_0[active] 
    # Matrice des distances entre tous les voxels
    # Terme des données
    ATA = A_sub.T @ (Cd_inv[:, None] * A_sub)   # A^T diag(Cd_inv) A
    # Second membre
    rhs = A_sub.T @ (Cd_inv * (data_obs - A_sub @ rho0_sub))

    model_grid = vtk.vtkStructuredGrid()
    model_grid.ShallowCopy(src_grid)
    
    mask_active = np.zeros(nvox)
    mask_active[active] = 1
    active_vtk = numpy_support.numpy_to_vtk(
            mask_active,
            deep=True,
            array_type=vtk.VTK_INT
        )
    active_vtk.SetName(f"active_voxels")
    model_grid.GetCellData().AddArray(active_vtk)
    
    weight_vtk = numpy_support.numpy_to_vtk(
            weight_concat.astype(np.float32),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
    weight_vtk.SetName(f"voxels_weight")
    model_grid.GetCellData().AddArray(weight_vtk)

    rays_intercept_vtk = numpy_support.numpy_to_vtk(
            rays_intercept.astype(np.int32),
            deep=True,
            array_type=vtk.VTK_INT
        )
    rays_intercept_vtk.SetName(f"rays_nintercept")
    model_grid.GetCellData().AddArray(rays_intercept_vtk)

    det_intercept_vtk = numpy_support.numpy_to_vtk(
            detectors_intercept.astype(np.int32),
            deep=True,
            array_type=vtk.VTK_INT
        )
    det_intercept_vtk.SetName(f"detectors_nintercept")
    model_grid.GetCellData().AddArray(det_intercept_vtk)

    rho0_vtk = numpy_support.numpy_to_vtk(
            D_0.astype(np.float32),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
    rho0_vtk.SetName(f"rho0")
    model_grid.GetCellData().AddArray(rho0_vtk)

    sensitivity_vtk = numpy_support.numpy_to_vtk(
            sensitivity.astype(np.float64),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
    sensitivity_vtk.SetName(f"sensitivity")
    model_grid.GetCellData().AddArray(sensitivity_vtk)

    str_regtype = "expdist_smooth"

    basename_out = input_file.stem

    file_out_npy = dir_inv / f"{basename_out}_{str_regtype}_{str_tel_type}.npz"

    nreg = 19
    # lambda_reg = np.logspace(-1, 1, nreg)

    models = np.zeros((nreg, nvox))
    regularizations = np.zeros((nreg, nvox))
    misfits_data, misfits_model = np.zeros(nreg), np.zeros(nreg)
    # sigma_rho = 0.3e3  # exemple 300 kg/m^3
    sigma_reg = np.logspace(0, 2, nreg)
    # lambda_reg = np.logspace(-3, -1, nreg)
    # sigma_reg = np.linspace(1e1, 9e2, nreg)

    l_corr = 100      # m
    dist_matrix = squareform(pdist(coords_active, metric='euclidean'))
    # print(dist_matrix.shape, dist_matrix[:100])
    exp_dist = np.exp(-dist_matrix / l_corr)

    rho_min, rho_max = min(D_1[active]),max(D_1[active])
    rho_min, rho_max = 1e3,4e3

    for i, sr in tqdm(enumerate(sigma_reg), total=nreg, desc='Modeling'):    
        # Covariance exponentielle
        C_rho = sr**2 * exp_dist
        # Inverse de C_rho
        # C_rho_inv = np.linalg.inv(C_rho)
        C_rho_inv = np.linalg.inv(C_rho + 1e-10 * np.eye(len(coords_active)))  # petite sécurité

        # Matrice totale
        M_total = ATA + C_rho_inv
        delta_rho = np.linalg.solve(M_total, rhs)
        rho_post = rho0_sub + delta_rho
        r_d = A_sub @ rho_post - data_obs
        chi2_d = np.sum((r_d / unc_obs)**2)
        misfits_data[i] = chi2_d
        r_m = C_rho_inv @ delta_rho
        chi2_m = delta_rho @ r_m
        misfits_model[i] = chi2_m
        rho_post = np.clip(rho_post, rho_min, rho_max)
        models[i, active] = rho_post
        min_post, max_post = np.min(rho_post), np.max(rho_post)
        mean_post = np.mean(rho_post)
        # print("mean post:", mean_post)
        # print(f"min, max post: {min_post}, {max_post}")
        print(f"{i}, {sr:.2e}: median=  {np.median(rho_post)}")
        rho_vtk = numpy_support.numpy_to_vtk(
            models[i].astype(np.float32),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        rho_vtk.SetName(f"{i}_density_sigma_{sr:.3e}")
        model_grid.GetCellData().AddArray(rho_vtk)
        reg_npy = np.zeros(nvox)
        reg_npy[active]= r_m.astype(np.float32)
        reg_vtk = numpy_support.numpy_to_vtk(
            reg_npy,
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        reg_vtk.SetName(f"{i}_reg_sigma_{sr:.3e}")
        model_grid.GetCellData().AddArray(reg_vtk)
        
    # active scalar par défaut
    model_grid.GetCellData().SetActiveScalars(
        f"{i}_density_sigma_{sigma_reg[0]:.3e}"
    )
    np.savez_compressed(file_out_npy,
                density_true = D_1, 
                density_post = models,
                misfits=np.asarray((misfits_data, misfits_model)),
                lambda_reg = sigma_reg,
                correlation_length = l_corr,
                mask_voxel = mask_active,
                rays_intercept=rays_intercept,
                sensitivity = sensitivity,
            )
    print(f"Saved npy {file_out_npy}")
    # '''
    basename_out = file_out_npy.stem
    # file_out_vts = dir_inv / f"{basename_out}_{kernel[:4]}reg_l{int(length)}m_lambda_series.vts"
    file_out_vts = dir_inv / f"{basename_out}.vts"
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(dir_inv / file_out_vts))
    writer.SetInputData(model_grid)
    writer.Write()
    print(f"Saved structured grid {file_out_vts}")