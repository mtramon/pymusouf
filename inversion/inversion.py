#!/usr/bin/python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from laplacian import build_laplacian_3d, build_graph_laplacian
from topography import build_interpolator_topography, interpolate_topography
from utils.tools import print_file_datetime

class Inversion:
    """
    Inversion déterministe linéaire :
        min || C_d^{-1/2} (M rho - d) ||^2
          + || C_m^{-1/2} (rho - rho0) ||^2
    avec matrices creuses et solveur itératif CG
    """

    def __init__(
        self,
        M: sp.csr_matrix,      # (ndata, nvox)
        data: np.ndarray,      # (ndata,)
        unc: np.ndarray,       # (ndata,)
        L: sp.csr_matrix,      # (nvox, nvox) régularisation (ex: Laplacien)
        sensitivity:np.ndarray, 
        diag_data:np.ndarray,
        weights:np.ndarray=None,
        diag_reg:np.ndarray=None,
    ):
        # --- checks ---
        assert sp.isspmatrix_csr(M)
        assert M.shape[0] == data.size
        assert data.size == unc.size
        assert L.shape[0] == L.shape[1] == M.shape[1]
        self.M = M
        self.data = data
        self.unc = unc
        self.Cd_inv = 1.0 / (unc**2)
        self.L = L
        self.ndata, self.nvox = M.shape
        self.sensitivity = sensitivity
        self.diag_data = diag_data
        self.weights = weights
        self.diag_reg = L.diagonal()
        # self.diag_reg = np.array(L.power(2).sum(axis=0))
        # self.diag_reg = self._compute_diag_LTWL(L, weights)

    def _compute_diag_LTWL(self, L, W):
        L_csc = L.tocsc()
        n = L.shape[0]
        diag = np.zeros(n)
        for j in range(n):
            start = L_csc.indptr[j]
            end = L_csc.indptr[j+1]
            rows = L_csc.indices[start:end]
            vals = L_csc.data[start:end]
            diag[j] = np.sum(W[rows] * vals**2)
        # print("_compute_diag_LTWL:", np.min(diag), np.max(diag))
        return diag
    
    def _build_rhs(self, rho0, lambda_reg, lambda_damp):
        b = self.M.T @ (self.Cd_inv * self.data)
        if rho0 is not None:
            b += lambda_reg * (self.L @ rho0)
            # b += lambda_reg * (self.L.T @ (self.L @ rho0))
            # 
            # Lrho0 = self.L @ rho0
            # WLrho0 = self.weights * Lrho0
            # b += lambda_reg * (self.L.T @ WLrho0)
            # Terme damping
            if lambda_damp != 0 :
                b += lambda_damp * rho0
        return b
    
    def _make_operator(self, lambda_reg, lambda_damp):

        def operator(rho):
            # Terme données
            y = self.M.T @ (self.Cd_inv * (self.M @ rho))
            # Terme régularisation pondéré
            # y += lambda_reg * (self.L.T @ (self.L @ rho)) #graph_laplacian
            y += lambda_reg * (self.L @ rho) 
            # Terme spatial
            # Lrho = self.L @ rho
            # WLrho = self.weights * Lrho   # multiplication par la diagonale W
            # y += lambda_reg * (self.L.T @ WLrho)
            # Terme damping
            if lambda_damp != 0 :
                y += lambda_damp * rho
            return y

        return LinearOperator(
            shape=(self.nvox, self.nvox),
            matvec=operator,
            dtype=np.float64
        )

    def _make_preconditioner(self, lambda_reg, lambda_damp):
        """
        Régularisation basée sur la senbilité de chaque voxel aux données
        sens = np.sqrt((M.power(2).T @ Cd_inv))
        """
        diag_A = self.diag_data +  lambda_reg * self.diag_reg 
        if lambda_damp >0 : diag_A += lambda_damp
        diag_A = np.maximum(diag_A, 1e-12)
        inv_diag = 1.0 / diag_A
        def precond(x):
            return inv_diag * x
        
        return LinearOperator(
            shape=(self.nvox, self.nvox),
            matvec=precond,
            dtype=np.float64
        )

    def misfit(self, rho, lambda_reg, lambda_damp):
        """
        Chi2 total (données + régularisation)
        """
        # terme données
        r_d = self.M @ rho - self.data
        chi2_d = np.sum((r_d / self.unc)**2)
        # terme modèle
        delta_rho = (rho - rho0)
        Lrho = self.L @ delta_rho
        chi2_m = lambda_reg * (Lrho@Lrho)
        # chi2_m = lambda_reg * np.sum(self.weights * (Lrho**2)) 
        # chi2_m += lambda_damp * (delta_rho @ delta_rho )
        return chi2_d, chi2_m


    def solve(
        self,
        rho0: np.ndarray,
        lambda_reg:float,
        lambda_damp: float = 0.0,
        tol: float = 1e-5,
        maxiter: int = 10000,
    ):
        """
        Résout le problème inverse par conjugate gradient (CG).
        """
        assert rho0.size == self.nvox
        self.rho0 = rho0
        P = self._make_preconditioner(lambda_reg, lambda_damp)
        A = self._make_operator(lambda_reg, lambda_damp)
        b = self._build_rhs(rho0, lambda_reg, lambda_damp)
        rho, info = cg(A, b, M=P, x0=rho0, rtol=tol, maxiter=maxiter)
        # rho, info = cg(A, b, x0=rho0, rtol=tol, maxiter=maxiter) #without preconditioner
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return rho


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
    mask_voi = voxel_density > 0.0   # True = voxel du volcan
    mask_voi = mask_voi.astype(np.uint8)  # Numba-friendly
    rho0 = 1800 # in kg/m^3
    D_0 = rho0 * np.ones(nvox, dtype=np.float64)  # density vector in kg/m^3
    D_0 = D_0 * mask_voi
    D_1 = voxel_density * mask_voi  
    if D_1.max() < 1e3: 
        D_1 *= 1e3 # if density is in g/cm^3, convert to kg/m^3
    dtel = CURRENT_SURVEY.telescope
    r, c = 0, 0
    data_concat = None
    M_concat = None
    detectors_intercept = np.zeros_like(mask_voi) 
    det_coords = [] 
    
    # basename = f"real_telescopes"   
    # dtel = CURRENT_SURVEY.telescope

    basename = f"toy_telescopes_s9506"   
    fin_toytel = dir_tel / f"toy_telescopes_s9506_vox{vs}m.pkl"
    with open(fin_toytel, 'rb') as f:
        dtel = pickle.load(f) 

    h5_path = dir_voxel / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"

    str_tel_type = basename.split("_")[0]

    with h5py.File(h5_path) as fh5 : 
            
        for tel_name, tel in tqdm(dtel.items(), desc="Data"):
            tel.compute_angular_coordinates()
            det_coords.append(tel.coordinates)
            # if tel.name == "SXF": continue
            # if tel.name == "OM" : continue
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
                rays_length = np.array(voxray["rays_length"])

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
                opacity[mask_rays] = M_norm.dot(D_1) [mask_rays] #/ G_uv [mask_rays]
                # print(np.mean(opacity[mask_rays]))
                mask_rays = mask_rays & (opacity > 1e3)
                mean_opacity, std_opacity = np.mean(opacity), np.std(opacity)
                # op_scaled = opacity[mask_rays]
                # op_scaled = (op_scaled-np.min(op_scaled)) / (np.max(op_scaled)-np.min(op_scaled))
                unc = np.zeros_like(opacity)
                # unc_rand = np.random.normal(0.05*opacity[mask_rays], 0.01*opacity[mask_rays])
                # unc[mask_rays] = np.clip(unc_rand, 1e-4*mean_opacity, 0.2*mean_opacity)
                # unc_model = 0.025 + 0.07/(1+np.exp(40*(op_scaled-0.02))) + 0.05/(1+np.exp(8*(0.9-op_scaled))) #noise model Lelièvre 2019
                # unc[mask_rays] = np.clip(unc_model*opacity[mask_rays], 1e-4*mean_opacity, 0.2*mean_opacity)
                # unc[mask_rays] = np.clip(unc_rand, 1e-4*mean_opacity, 0.2*mean_opacity)
                unc[mask_rays] = np.clip(0.02*opacity[mask_rays], 1e-4*mean_opacity, 0.2*mean_opacity)
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
    ORDER = "F"
    mask_center = (R <= 500).reshape(-1,order=ORDER)
    mask_z = (Z > abs(zmax-300)).reshape(-1,order=ORDER)
    weight_concat = np.array(M_concat.sum(axis=0)).reshape(-1,order=ORDER)
    mask_weight = (weight_concat > 0)
    rays_intercept = np.diff(M_concat.tocsc().indptr)
    # mask_rays_intercept = (rays_intercept <= np.percentile(rays_intercept, 99.9)) 
    mask_det_intercept = (detectors_intercept >= 3)  
    opaque = np.where(data_concat > 0 )[0]
    data_concat = data_concat[opaque]
    unc_concat = unc_concat[opaque]
    print("data: min, max: ", np.min(data_concat), np.max(data_concat))
    print("unc: min, max: ",np.min(unc_concat), np.max(unc_concat))

    mask_voxel = mask_voi & mask_center & mask_z & mask_det_intercept & mask_weight
    coords = np.vstack((X.ravel(order=ORDER), Y.ravel(order=ORDER), Z.ravel(order=ORDER))).T
   
    sensitivity = np.zeros(nvox)
    diag_data = np.zeros(nvox)
    
    _active = np.where(mask_voxel)[0]
    _M = M_concat[opaque][:, _active]
    print("VOI : nrays, nvoxels = ", _M.shape)
    
    # diag(M^T C_d^-1 M) = # diag(M^T W M)
    Cd_inv = 1 / (unc_concat**2)
    _diag_data = np.array(_M.power(2).T @ Cd_inv).ravel(order=ORDER)
    sensitivity[_active] = np.sqrt(_diag_data)
    
    mask_voxel = mask_voxel & (sensitivity>0).ravel(order=ORDER)    
    active = np.where(mask_voxel)[0]    
    M_concat = M_concat[opaque][:, active]
    diag_data[active] = sensitivity[active]**2
    scale_sens = np.median(sensitivity[active])
    # M_concat = M_concat / scale_sens
    # diag_data /= scale_sens**2
    # sensitivity /= scale_sens
    nvox_active = len(active)

    # rescale sensibilité
    # M_concat = M_concat / np.sqrt(scale_sens)
    # diag_data = diag_data / scale_sens
    # sensitivity[active] /= np.sqrt(scale_sens)
    #
    
    str_regtype = "graph_laplacian_gausdist"

    # interp = build_interpolator_topography(file=dir_dem / "topo_roi.vts")
    # Z_interp = interpolate_topography(interp, X, Y).ravel(order=ORDER)[active] #
    # Z_interp = interp(X[:,:,0].ravel(), Y[:,:,0].ravel())
    # z_coords = coords[active,2]
    # depth = np.maximum(0.0, Z_interp - z_coords)
    # print(depth.shape)
    # print("depth:",np.min(depth), np.max(depth))
    # exit()
    # alpha, z0 = 5.0, 50
    # print(depth[:100])
    # weights = 1.0 + alpha * np.exp(- depth / z0)


    # L = build_laplacian_3d(nvx, nvy, nvz)
    # L = L[active][:, active]
    # diag_LtL = np.array(L.power(2).sum(axis=0)).ravel(ORDER)
    # scale_L = np.median(diag_LtL)
    # L = L / np.sqrt(scale_L)
    
    
    # fig, ax = plt.subplots()
    # x = np.linspace(-vs, 300, 100)
    # y = 1.0 + alpha * np.exp(- x / z0)
    # ax.plot(x, y)
    # ax.scatter(depth,weights )
    # fig.savefig("depth.png")
    # print(np.all(weights>0))

    length = 1
    kernel =  "gaussian" 
    r_cut = np.sqrt(3)*vs
    # r_cut = 10*vs
    L = build_graph_laplacian(coords=coords[active], length=length, r_cut=r_cut, kernel=kernel) 
    L = L + 1e-6 * sp.eye(L.shape[0])
    # diag_LtL = np.array(L.power(2).sum(axis=0)).ravel(ORDER)
    # scale_L = scale_sens**2 / np.median(L.diagonal())
    # L = L / scale_L
    # fout_lreg = dir_inv / f"graph_laplacian_{kernel[:4]}_l{int(length)}m.npz"
    # np.savez_compressed(fout_lreg, data=L.data, indptr=L.indptr, indices=L.indices, shape=L.shape)
    # print(f"Save {fout_lreg}")

    # weights = 1.0 / (sensitivity[active] + 1e-8)
    # weights = np.ones_like(active, dtype=np.float64)

    # det_coords = np.asarray(det_coords)
    # det_tree = cKDTree(det_coords)
    # d, _ = det_tree.query(coords[active])  # coords_active = coordonnées des voxels actifs
    # dmax = 1e3
    # print(d.shape)
    # print("distances:",d.shape, coords[active].shape)
    # d0 = 100 #np.std(distances)
    # alpha, p = 1, 2
    # weights = 1 + alpha * (d/dmax)**p 
    # weights = 1 + alpha * np.exp(distances / d0)
    # weights = 1 + alpha *  distances**2  
    # weights /= np.mean(weights)

    # print("weights:",weights[:100])
    # diag_reg = np.array(L.diagonal()).copy()  # déjà un array

    # weights = np.maximum(weights, 1e-12)
    # print(weights.shape)
    print("is weights all pos :", np.all(weights>0),  weights[:10])
    print("is diag_data all pos :", np.all(diag_data>0), diag_data[:10])
    # print("is diag_reg all pos :", np.all(diag_reg>0))
    # exit()
    # weights = np.ones(nvox_active)
    # print(weights.shape)
    inv = Inversion(
        M=M_concat,
        data=data_concat,
        unc=unc_concat,
        L=L,
        sensitivity=sensitivity[active],
        diag_data=diag_data[active],
        # diag_reg=diag_reg,
        weights=weights,
    )
   
    model_grid = vtk.vtkStructuredGrid()
    model_grid.ShallowCopy(src_grid)
    
    mask_sensitive = np.zeros(nvox)
    mask_sensitive[active] = 1
    active_vtk = numpy_support.numpy_to_vtk(
            mask_sensitive.ravel(),
            deep=True,
            array_type=vtk.VTK_INT
        )
    active_vtk.SetName(f"active_voxels")
    model_grid.GetCellData().AddArray(active_vtk)
    
    # depth_npy = np.zeros(nvox)
    # depth_npy[active]=depth
    # depth_vtk = numpy_support.numpy_to_vtk(
    #         depth_npy,
    #         deep=True,
    #         array_type=vtk.VTK_INT
    #     )
    # depth_vtk.SetName(f"depth_voxels")
    # model_grid.GetCellData().AddArray(depth_vtk)

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
    
    sensitivity_vtk = numpy_support.numpy_to_vtk(
            sensitivity.astype(np.float64),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
    sensitivity_vtk.SetName(f"sensitivity")
    model_grid.GetCellData().AddArray(sensitivity_vtk)
    
    
    # reg0 = np.zeros(nvox)
    # diag_reg = inv.diag_reg
    # reg0[active] = diag_reg * D_0[active]
    # reg0_vtk = numpy_support.numpy_to_vtk(
    #         reg0.astype(np.float64),
    #         deep=True,
    #         array_type=vtk.VTK_FLOAT
    #     )
    # reg0_vtk.SetName(f"reg0")
    # model_grid.GetCellData().AddArray(reg0_vtk)

    det_intercept_vtk = numpy_support.numpy_to_vtk(
            detectors_intercept.astype(np.int32),
            deep=True,
            array_type=vtk.VTK_INT

        )
    det_intercept_vtk.SetName(f"detectors_nintercept")
    model_grid.GetCellData().AddArray(det_intercept_vtk)

    # file_out_npy = dir_inv / f"density_models_{kernel[:4]}reg_l{int(length)}m_vox{vs}m.npz"
    file_out_npy = dir_inv / f"{input_file.stem}_{str_regtype}_{str_tel_type}.npz"
    # '''

    nreg = 30
    # lambda_reg = np.logspace(3, 5, nreg)
    # lambda_reg = np.logspace(-1, 2, nreg)
    lambda_reg = np.logspace(1, 4, nreg)
    models = np.zeros((nreg, nvox))
    regularizations = np.zeros((nreg, nvox))
    misfits_data, misfits_model = np.zeros(nreg), np.zeros(nreg)
    
    ld=0
    for i, lr in tqdm(enumerate(lambda_reg), total=nreg, desc='Modeling'):  
        # print(D_0[active])  
        # ld = 1e-6*lr
        rho_post = inv.solve(rho0=D_0[active], lambda_reg=lr, lambda_damp=ld)
        # rho_post *= scale_sens
        misfits_data[i], misfits_model[i] = inv.misfit(rho_post, lr, ld)
        models[i, active] = rho_post
        min_post, max_post = np.min(rho_post), np.max(rho_post)
        mean_post = np.mean(rho_post)
        print("mean post:", mean_post)
        print(f"min, max post: {min_post}, {max_post}")
        print(f"{i}, {lr:.2e}: median=  {np.median(rho_post)}")
        # rho_order = models[i].reshape((nvx, nvy, nvz), order=ORDER)
        # rho_order = np.transpose(rho_order, (2, 1, 0)).ravel(order=ORDER)
        rho_vtk = numpy_support.numpy_to_vtk(
            models[i],
            deep=True,
            array_type=vtk.VTK_FLOAT, 
        )
        rho_vtk.SetName(f"{i}_density_lambda_{lr:.3e}")
        model_grid.GetCellData().AddArray(rho_vtk)
        # regularizations[i, active] = L.T @ (L @ rho_post)
        # regularizations[i, active] = L @ rho_post
        # reg_vtk = numpy_support.numpy_to_vtk(
        #     regularizations[i],
        #     deep=True,
        #     array_type=vtk.VTK_FLOAT
        # )
        # reg_vtk.SetName(f"i}_reg_lambda_{lr:.3e}")
        # model_grid.GetCellData().AddArray(reg_vtk)
    # active scalar par défaut
    model_grid.GetCellData().SetActiveScalars(
        f"density_lambda_{lambda_reg[0]:.3e}"
    )
    np.savez_compressed(file_out_npy,
                density_true = D_1, 
                density_post = models,
                misfits=np.asarray((misfits_data, misfits_model)),
                lambda_reg = lambda_reg,
                sensitivity = sensitivity,
                mask_voxel = mask_sensitive,
                rays_intercept=rays_intercept,
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
    
   

    
    