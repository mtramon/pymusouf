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
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime, check_array_order


def global_to_local(nx, ny, nz, mask):
    """
    Crée un tableau de mapping indice global -> indice local.

    Paramètres
    ----------
    nx, ny, nz : int
        Dimensions de la grille complète (nombre de voxels dans chaque direction).
    mask : array bool 1D de taille nx*ny*nz
        True pour les voxels actifs, False pour les inactifs.

    Retourne
    --------
    local_idx : array int de même taille que mask
        Pour chaque indice global (dans l'ordre Fortran, i + j*nx + k*nx*ny),
        local_idx[glob] vaut :
            - l'indice local (0 .. N_active-1) si mask[glob] est True,
            - -1 sinon.
    """
    n_global = nx * ny * nz
    assert mask.size == n_global, "La taille du masque ne correspond pas à nx*ny*nz"
    local_idx = np.full(n_global, -1, dtype=int)
    local_idx[mask.astype(bool)] = np.arange(np.count_nonzero(mask))
    return local_idx


class InversionTV:
    """
    Inversion avec régularisation Variation Totale (isotrope) + damping optionnel.
    Résolution par FISTA avec proximal de Chambolle.
    """
    def __init__(self, M, data, unc, nx, ny, nz, mask, 
                 rho0=None, 
                 sensitivity=None, 
                 weights=None, 
                 weights_z=None):
        """
        M : matrice d'observation (creuse CSR), shape (ndata, nvox)
        data : vecteur des données (ndata,)
        unc : incertitudes (ndata,) -> Cd^-1 = diag(1/unc**2)
        nx, ny, nz : dimensions du volume
        lambda_reg : poids de la régularisation TV
        mu : poids du damping (0 par défaut)
        rho0 : modèle a priori pour le damping (si mu>0)
        """
        self.M = M
        self.data = data
        self.unc = unc
        self.Cd_inv = 1.0 / (unc**2 + 1e-12)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nvox_total = nx * ny * nz
        self.mask = mask
        self.nvox = M.shape[1]  
        self.weights = weights
        self.weights_z = weights_z
        self.rho0 = rho0 if rho0 is not None else np.zeros(self.nvox)
        self.sensitivity = sensitivity
        # Construction des opérateurs de gradient (différences avant) pour la TV
        self.Dx, self.Dy, self.Dz = self._build_gradient_ops(mask)


    def _build_gradient_ops(self, mask):
        nx, ny, nz = self.nx, self.ny, self.nz
        N = np.count_nonzero(mask)
        local_idx = global_to_local(nx, ny, nz, mask)

        # Listes pour la construction des matrices au format COO
        rows_x, cols_x, vals_x = [], [], []
        rows_y, cols_y, vals_y = [], [], []
        rows_z, cols_z, vals_z = [], [], []

        # Diagonale : -1 pour tous les voxels actifs (base de la différence)
        for loc in range(N):
            rows_x.append(loc); cols_x.append(loc); vals_x.append(-1.0)
            rows_y.append(loc); cols_y.append(loc); vals_y.append(-1.0)
            rows_z.append(loc); cols_z.append(loc); vals_z.append(-1.0)

        # Parcours des voxels pour ajouter les contributions des voisins
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    glob = i + j*nx + k*nx*ny
                    if not mask[glob]:
                        continue
                    loc = local_idx[glob]

                    # Voisin en x positif (i+1)
                    if i + 1 < nx:
                        glob_n = (i+1) + j*nx + k*nx*ny
                        if mask[glob_n]:
                            loc_n = local_idx[glob_n]
                            rows_x.append(loc)
                            cols_x.append(loc_n)
                            vals_x.append(1.0)

                    # Voisin en y positif (j+1)
                    if j + 1 < ny:
                        glob_n = i + (j+1)*nx + k*nx*ny
                        if mask[glob_n]:
                            loc_n = local_idx[glob_n]
                            rows_y.append(loc)
                            cols_y.append(loc_n)
                            vals_y.append(1.0)

                    # Voisin en z positif (k+1)
                    if k + 1 < nz:
                        glob_n = i + j*nx + (k+1)*nx*ny
                        if mask[glob_n]:
                            loc_n = local_idx[glob_n]
                            rows_z.append(loc)
                            cols_z.append(loc_n)
                            vals_z.append(1.0)

        # Création des matrices CSR
        Dx = sp.coo_matrix((vals_x, (rows_x, cols_x)), shape=(N, N)).tocsr()
        Dy = sp.coo_matrix((vals_y, (rows_y, cols_y)), shape=(N, N)).tocsr()
        Dz = sp.coo_matrix((vals_z, (rows_z, cols_z)), shape=(N, N)).tocsr()

        return Dx, Dy, Dz

    def _estimate_lipschitz(self, power_iter=20, mu=0, mu_z=0):
        """Estime la plus grande valeur propre de M^T Cd^{-1} M + mu I par puissance itérée."""
        n = self.nvox
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        for _ in range(power_iter):
            y = self.M.T @ (self.Cd_inv * (self.M @ x))
            if mu > 0:
                y += mu * x
            ##Damping vertical
            if mu_z > 0:
                # Terme vertical : tau * Dz.T @ (weight_z * (Dz @ x))
                Dzx = self.Dz @ x
                wDzx = self.weights * Dzx
                y += mu_z * (self.Dz.T @ wDzx)
            ##
            norm_y = np.linalg.norm(y)
            x = y / norm_y
        return norm_y

    def _prox_tv_chambolle_iso(self, v, lambda_tv, n_iter=20):
        """
        Proximal de la TV isotrope par l'algorithme de Chambolle.
        v : vecteur (nvox,)
        lambda_tv : paramètre (déjà multiplié par le pas)
        Retourne u = prox_{lambda_tv * TV}(v)
        """
        n = self.nvox
        # Initialiser les variables duales p, q, r (pour chaque direction)
        p = np.zeros((n, 3))
        tau = 0.25  # pas pour la convergence

        for _ in range(n_iter):
            # Divergence de p
            div_p = self.Dx.T @ p[:,0] + self.Dy.T @ p[:,1] + self.Dz.T @ p[:,2]
            u = v - lambda_tv * div_p
            # Mise à jour de p par projection sur la boule unité
            Gx = self.Dx @ u
            Gy = self.Dy @ u
            Gz = self.Dz @ u
            nrm = np.sqrt(Gx**2 + Gy**2 + Gz**2) + 1e-12
            # nrm = self.weights * np.sqrt(Gx**2 + Gy**2 + Gz**2)
            p[:,0] = (p[:,0] + tau * Gx) / (1 + tau * nrm)
            p[:,1] = (p[:,1] + tau * Gy) / (1 + tau * nrm)
            p[:,2] = (p[:,2] + tau * Gz) / (1 + tau * nrm)
        return u
    
    def _prox_tv_chambolle_aniso(self, v, lambda_tv, n_iter=20, lambda_x=1, lambda_y=1, lambda_z=10):
        """
        Proximal de la TV anisotrope par l'algorithme de Chambolle.
        v : vecteur (nvox,)
        lambda_tv : paramètre (déjà multiplié par le pas)
        Retourne u = prox_{lambda_tv * TV}(v)
        """
        n = self.nvox
        # Initialiser les variables duales p, q, r (pour chaque direction)
        p = np.zeros((n, 3))
        tau = 0.25  # pas pour la convergence

        for _ in range(n_iter):
            # Divergence de p
            div_p = self.Dx.T @ p[:,0] + self.Dy.T @ p[:,1] + self.Dz.T @ p[:,2]
            u = v - lambda_tv * div_p
            # Mise à jour de p par projection sur la boule unité
            Gx = self.Dx @ u
            Gy = self.Dy @ u
            Gz = self.Dz @ u
            nrm = np.sqrt(
                lambda_x * Gx**2 +
                lambda_y * Gy**2 +
                lambda_z * Gz**2
            )            + 1e-12
            
            p[:,0] = (p[:,0] + tau * Gx) / (1 + tau * nrm)
            p[:,1] = (p[:,1] + tau * Gy) / (1 + tau * nrm)
            p[:,2] = (p[:,2] + tau * Gz) / (1 + tau * nrm)
        return u

    def solve(self, rho_init=None, rho_range=[1e3, 4e3], lambda_reg=None, mu=0, mu_z=0, max_iter=500, tol=1e-5,  verbose=False):
        """
        Résout le problème par FISTA.
        """
        if rho_init is None:
            rho = np.zeros(self.nvox)
        else:
            rho = rho_init.copy()

        y = rho.copy()
        t = 1.0
        # Constante de Lipschitz de la partie lisse
        self.Lf = self._estimate_lipschitz(mu=mu, mu_z=mu_z)
        gamma = 1.0 / self.Lf  # pas
        rho_min, rho_max = rho_range
        obj_prev = np.inf
        for i in range(max_iter):
            # Gradient de la partie lisse en y
            grad = self.M.T @ (self.Cd_inv * (self.M @ y - self.data))
            if mu > 0:
                grad += mu * (y - self.rho0)
                if self.weights is not None: grad *= self.weights
          
            # régularisation verticale
            if mu_z > 0: 
                Dz_y = self.Dz @ y
                if self.weights_z is not None:   grad += mu_z * (self.Dz.T @ (self.weights_z * Dz_y))
            
            # Descente de gradient
            x = y - gamma * grad

            # Proximal de la TV
            rho_new = self._prox_tv_chambolle_iso(x, lambda_reg * gamma)


            rho_new = np.clip(rho_new, rho_min, rho_max)
            # Mise à jour de FISTA
            t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
            y = rho_new + (t-1)/t_new * (rho_new - rho)
            # Calcul de l'objectif (optionnel)
            if verbose and i % 10 == 0:
                obj = self._objective(rho_new)
                print(f"It {i}, obj = {obj:.3e}")
            # Test de convergence
            diff = np.linalg.norm(rho_new - rho)
            if diff < tol:
                if verbose:
                    print(f"Converged at iteration {i}")
                rho = rho_new
                break
            rho, t = rho_new, t_new
        return rho

    def _objective(self, rho, lambda_reg, mu=0, mu_z=0):
        """Calcule la fonction coût totale."""
        # Terme données
        r = self.M @ rho - self.data
        data_term = 0.5 * np.sum(r**2 * self.Cd_inv)
        # Terme TV
        Gx = self.Dx @ rho
        Gy = self.Dy @ rho
        Gz = self.Dz @ rho
        tv_term = lambda_reg * np.sum(np.sqrt(Gx**2 + Gy**2 + Gz**2))
        # Damping
        damp_term = 0
        if mu > 0:
            if self.weights is not None : mu = mu * self.weights 
            damp_term = 0.5 * np.sum(mu * (rho - self.rho0)**2)
        return data_term + tv_term + damp_term

    def misfit(self, rho, lambda_reg, mu, mu_z=0):
        """
        Calcule les misfits pour un modèle rho donné.
        Retourne un tuple (chi2_data, tv_term, damp_term) où :
            chi2_data = (M rho - data)^T Cd^{-1} (M rho - data)
            tv_term = lambda_reg * sum sqrt((Dx rho)^2 + (Dy rho)^2 + (Dz rho)^2)
            damp_term = (mu/2) * ||rho - rho0||^2 (si mu>0)
        """
        r = self.M @ rho - self.data
        chi2_data = np.sum(r**2 * self.Cd_inv)

        Gx = self.Dx @ rho
        Gy = self.Dy @ rho
        Gz = self.Dz @ rho
        tv_term = lambda_reg * np.sum(np.sqrt(Gx**2 + Gy**2 + Gz**2))
        damp_term = 0.0
        if mu > 0:
            diff = rho - self.rho0
            if self.weights is not None : mu = mu * self.weights 
            damp_term = 0.5  * np.sum(mu * diff**2)
        return chi2_data, tv_term, damp_term


if __name__ == "__main__":

    from topography import build_interpolator_topography, interpolate_topography
    from voxelgrid import load_voxel_grid

    survey_name = CURRENT_SURVEY.name   
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"
    dir_tel = dir_survey / "telescope"
    dir_inv = dir_model / "inversion"
    dir_inv.mkdir(parents=True, exist_ok=True)
    vs = int(sys.argv[1]) if len(sys.argv) >1 else 32  # voxel size in m (edge length)
    # input_vtk = dir_model / f"ElecCond_topo_voi_vox{vs}m.vts"

    input_vtk = dir_voxel / f"topo_center_anom_voi_vox{vs}m.vts"
    print_file_datetime(input_vtk)
    
    grid, geom = load_voxel_grid(input_vtk)
    mask_voi = geom.mask_voxel
    print(np.count_nonzero(mask_voi))
    voxel_density  = geom.density
    nvox = len(mask_voi)
    rho0 = 1800 # in kg/m^3
    D_0 = rho0 * np.ones(nvox, dtype=np.float64)  # density vector in kg/m^3
    D_0 = D_0 * mask_voi
    D_1 = voxel_density * mask_voi 
    # print(check_array_order(D_1) )
    if D_1.max() < 1e3: 
        D_1 *= 1e3 # if density is in g/cm^3, convert to kg/m^3
    dtel = CURRENT_SURVEY.telescope
    r, c = 0, 0
    data_concat = None
    M_concat = None
    detectors_intercept = np.zeros_like(mask_voi) 
    det_coords = [] 

    basename = f"real_telescopes"   
    dtel = CURRENT_SURVEY.telescope

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
            x0, y0, z0 = tel.coordinates
            for conf_name, conf in tel.configurations.items():
                ze_matrix = tel.zenith_matrix[conf_name]
                mask_rays = (ze_matrix <= 90 * np.pi/180).ravel()
                nu, nv = conf.shape_uv
                u_edges, v_edges = conf.u_edges, conf.v_edges
                u, v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
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
                # M_norm = M.multiply(1.0 / (rays_length[:, None] * G_uv[:, None]))
                M_norm = M.multiply(1.0 / G_uv[:, None])
                # print(f"{tel_name}, {conf_name}: min, max M_norm.sum( axis=1)) = {np.min(M_norm.sum( axis=1)):.3e}, {np.max(M_norm.sum( axis=1)):.3e}")
                opacity[mask_rays] = M_norm.dot(D_1) [mask_rays] #/ G_uv [mask_rays]
                mask_rays = mask_rays & (opacity > 1e3)
                mean_opacity, std_opacity = np.mean(opacity), np.std(opacity)
                unc = np.zeros_like(opacity)
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
    opaque = np.where(data_concat > 0 )[0]
    ndata = len(opaque)
    data_concat = data_concat[opaque]
    unc_concat = unc_concat[opaque]
    print("data: min, max: ", np.min(data_concat), np.max(data_concat))
    print("unc: min, max: ",np.min(unc_concat), np.max(unc_concat))
    
    mask_voxel = mask_voi & mask_center & mask_z & mask_det_intercept & mask_weight #& mask_rays_intercept
    # print(check_array_order(mask_voxel))
    
    # print(np.count_nonzero(mask_voxel))
    coords = np.vstack((X.ravel(order=ORDER), Y.ravel(order=ORDER), Z.ravel(order=ORDER))).T
    # print(check_array_order(coords))

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
    print(sthresh)
    mask_voxel = mask_voxel #& (sensitivity>sthresh)    
    active = np.where(mask_voxel)[0]    
    sens_active = sensitivity[active]

    M_concat = M_concat[opaque][:, active]
    nvox_active = len(active)
    str_reg_type = "tv"


    #### PONDERATION DES VOXELS
    det_coords = np.asarray(det_coords)
    det_tree = cKDTree(det_coords)
    d, _ = det_tree.query(coords[active])
    dist_min = min(d)#1e2

    # d0 = 10*vs
    # wd = 1 - np.exp(-((d-dist_min)/d0)**2)
   
    file_dem = dir_dem / "topo_roi.vts"
    interp = build_interpolator_topography(file=file_dem)
    Z_interp = interpolate_topography(interp, X, Y).ravel(ORDER)[active] #
    z_coords = coords[active,2]
    depth = np.maximum(0.0, Z_interp - z_coords)

    
    wd = ((d - dist_min) / (d.max() - dist_min + 1e-8))**2
    wd = np.clip(wd, 0, 1)

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
    weights = wd * (1 + gamma * ws)

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
    # weights = wd

    ####
    wz = np.zeros_like(s)

    s = sensitivity[_active]
    s = s / np.max(s)
    sthresh = np.percentile(sensitivity[_active], 50)
    beta = 2
    mask_slow = (s < sthresh)
    wz[mask_slow] = (1 - s[mask_slow])**beta

    # normalisation (important)
    # wz = wz / (wz.max() + 1e-8)
    wz = None

    rho_1 = np.median(D_1[active])
   
    tv = InversionTV(
        M=M_concat,
        data=data_concat,
        unc=unc_concat,
        nx=nvx, ny=nvy, nz=nvz,
        mask=mask_voxel,
        weights=weights,
        weights_z = wz,
        rho0=np.ones(nvox_active) * rho_1, 
        sensitivity=sens_active,
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

    # depth_npy = np.zeros(nvox)
    # depth_npy[active]=depth
    # depth_vtk = numpy_support.numpy_to_vtk(
    #         depth_npy,
    #         deep=True,
    #         array_type=vtk.VTK_INT
    #     )
    # depth_vtk.SetName(f"depth_voxels")
    # model_grid.GetCellData().AddArray(depth_vtk)

    file_out_npy = dir_inv / f"{input_vtk.stem}_{str_reg_type}_{str_tel_type}.npz"
    # '''

    nlr, nmu = 10, 1
    lambda_reg = np.logspace(-4, -1, nlr)
    mu_damp = np.logspace(-2, -1, nmu)
    ntot = nmu * nlr
    models = np.zeros((ntot, nvox))
    regularizations = np.zeros((ntot, nvox))
    misfits_data, misfits_model = np.zeros(ntot), np.zeros(ntot)
    c = 0 
    mu, mu_z = 0, 0.99
    rho_min, rho_max = min(D_1[active]), max(D_1[active])
    rho_init = np.ones(active.shape)*rho_1
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
                density_true = D_1, 
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
    basename_out = file_out_npy.stem
    # file_out_vts = dir_inv / f"{basename_out}_{kernel[:4]}reg_l{int(length)}m_lambda_series.vts"
    file_out_vts = dir_inv / f"{basename_out}.vts"
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(dir_inv / file_out_vts))
    writer.SetInputData(model_grid)
    writer.Write()
    print(f"Saved structured grid {file_out_vts}")
    
   

    