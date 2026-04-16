#!/usr/bin/python3
# -*- coding: utf-8 -*-

import h5py
import numba
import numpy as np
import matplotlib.pyplot as plt
# import pickle
import scipy.sparse as sp
import sys
from tqdm import tqdm
import vtk
# from utils.functions import smooth_window

# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
try: from .voxelgrid import load_voxel_grid
except: from voxelgrid import load_voxel_grid
# from telescope import DICT_TEL


titlesize="xx-large"
fontsize="xx-large"
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
         'figure.figsize': (16,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
plt.rcParams.update(params)


def angular_acceptance_map(
        u,
        v,
        delta_z,
        Lx=800.0,
        Ly=800.0,
    ):
        """
        Acceptance géométrique normalisée A(u,v)
        selon Thomas & Willis (rectangles).

        u, v : centres bins (dx/dz, dy/dz) ou (tan(theta_x), tan(theta_y))
        delta_z : séparation entre plans extrêmes (mm)
        Lx, Ly : dimensions du panel (mm)
        """
        # centres des bins
        U, V = np.meshgrid(u, v, indexing="ij")
        Ax = np.maximum(0.0, 1.0 - np.abs(U) * delta_z / Lx)
        Ay = np.maximum(0.0, 1.0 - np.abs(V) * delta_z / Ly)
        A = Ax * Ay
        return A

def plot_acceptance_map(u, v, acceptance):
    fig, ax = plt.subplots()
    im = ax.imshow(acceptance.T, origin="lower", extent=[u[0], u[-1], v[0], v[-1]], aspect="auto")
    ax.set_xlabel("tan(theta_x)")
    ax.set_ylabel("tan(theta_y)")   
    fig.colorbar(im, label="Acceptance")
    plt.show()

def set_rays_with_acceptance_midpoint(
    tel,
    u_edges,
    v_edges,
    delta_z=1.2,
    Lx=0.8,
    Ly=0.8, 
    n_sr=5,
):
    """
    Génère des sous-rayons intégrant l'acceptance du détecteur.

    Paramètres
    ----------
    u_edges, v_edges : array (Nbins+1)
        Bords angulaires des pixels (tan(theta_x), tan(theta_y))

    delta_z : float
        Séparation entre plans extrêmes (m)    
    Lx, Ly : float 
        Dimensions du panel (m)
    nu_sr, nv_sr : int
        Sous-échantillonnage angulaire par pixel

    Retour
    ------
    rays_dirs : (Nrays, 3)
    rays_weights : (Nrays,)
    rays_pixel_ids : (Nrays,)
    rays_is_main : (Nrays,)

    """
    rays_dirs = []
    rays_weights = []
    rays_pixel_ids = []
    rays_is_main = []
    nu_pix = len(u_edges) - 1
    nv_pix = len(v_edges) - 1
    G_r = np.zeros(nu_pix*nv_pix, dtype=np.float64)
    R = tel.rotation_matrix # rotation du télescope pour passer de la direction locale (u,v,1) à la direction globale
    # eps= 0.02
    for j in range(nv_pix):
        for i in range(nu_pix):
            u0, u1 = u_edges[i], u_edges[i+1]
            v0, v1 = v_edges[j], v_edges[j+1]
            du = (u1 - u0) / n_sr
            dv = (v1 - v0) / n_sr
            # sous-bins réguliers (midpoint rule)
            usr = u0 + (np.arange(n_sr) + 0.5) * du
            vsr = v0 + (np.arange(n_sr) + 0.5) * dv
            pixel_id = i * nv_pix + j
            
            for v in vsr:
                for u in usr:
                    # Direction locale (x, y, z)
                    dir_local = np.array([
                        - u ,
                        - v ,
                        1.0 ,
                    ], dtype=float)
                    # dir_local /= np.sqrt(1+u**2+v**2)
                    # acceptance locale (évaluée analytiquement)
                    S = max(0, Lx - abs(u)*delta_z) \
                        * max(0, Ly - abs(v)*delta_z)
                    dG = S * du * dv / (1 + u**2 + v**2)**2 # S * cos(theta) * dOmega
                    if dG <= 0.0:
                        continue
                    # Direction globale
                    dir_global =  R @ dir_local
                    rays_dirs.append(dir_global)
                    rays_weights.append(dG)
                    rays_pixel_ids.append(pixel_id)
                    is_main_ray = (u == usr[n_sr//2] and v == vsr[n_sr//2])  # le rayon central du pixel
                    rays_is_main.append(is_main_ray)
                    G_r[pixel_id] += dG
    return (
        np.asarray(rays_dirs, dtype=np.float64),
        np.asarray(rays_weights, dtype=np.float64),
        np.asarray(rays_pixel_ids, dtype=np.int64),
        np.asarray(rays_is_main, dtype=bool),
        G_r
    )


def set_rays_with_acceptance_rand(
    u_edges,
    v_edges,
    delta_z=1.2,
    Lx=0.8,
    Ly=0.8,
    n_sr=50,        # nombre total de sous-rayons par pixel
    seed = 9506
):
    """
    Génération isotrope de sous-rayons dans chaque pixel
    (uniforme en angle solide).
    """
    rays_dirs = []
    rays_weights = []
    rays_pixel_ids = []
    rays_is_main = []
    nu_pix = len(u_edges) - 1
    nv_pix = len(v_edges) - 1
    G_r = np.zeros(nu_pix * nv_pix, dtype=np.float64)
    R = tel.rotation_matrix
    rng = np.random.default_rng(seed)
    u_max = max(abs(u_edges[0]), abs(u_edges[-1]))
    v_max = max(abs(v_edges[0]), abs(v_edges[-1]))
    for j in range(nv_pix):
        for i in range(nu_pix):
            u0, u1 = u_edges[i], u_edges[i+1]
            v0, v1 = v_edges[j], v_edges[j+1]

            pixel_id = i * nv_pix + j
            # Aire du pixel en (u,v)
            du_pix = u1 - u0
            dv_pix = v1 - v0
            # échantillonnage aléatoire uniforme dans le pixel
            u_samples = rng.uniform(u0, u1, n_sr)
            v_samples = rng.uniform(v0, v1, n_sr)
            for k in range(n_sr):
                u = u_samples[k]
                v = v_samples[k]
                # direction locale normalisée
                norm = 1#np.sqrt(1 + u*u + v*v)
                dir_local = np.array([
                    -u / norm,
                    -v / norm,
                     1.0 / norm
                ])
                # acceptance géométrique
                Sphys = max(0, Lx - abs(u)*delta_z) \
                   * max(0, Ly - abs(v)*delta_z)
                # width_frac = 0.05
                # Staper = smooth_window(u, u_max, width_frac=width_frac) * smooth_window(v, v_max,width_frac=width_frac)
                S = Sphys
                if S <= 0:
                    continue
                # élément d'angle solide correspondant
                dOmega = (du_pix * dv_pix) / n_sr \
                         / (1 + u*u + v*v)**(3/2)
                cos_theta = 1 / np.sqrt(1 + u*u + v*v)
                dG = S * dOmega * cos_theta
                dir_global = R @ dir_local
                rays_dirs.append(dir_global)
                rays_weights.append(dG)
                rays_pixel_ids.append(pixel_id)
                # on marque comme "main ray" le rayon le plus proche du centre
                if k == 0:
                    rays_is_main.append(True)
                else:
                    rays_is_main.append(False)
                G_r[pixel_id] += dG

    return (
        np.asarray(rays_dirs),
        np.asarray(rays_weights),
        np.asarray(rays_pixel_ids),
        np.asarray(rays_is_main),
        G_r
    )


def set_rays_with_acceptance_jitter(
    u_edges,
    v_edges,
    delta_z=1.2,
    Lx=0.8,
    Ly=0.8,
    nu_sr=5,
    nv_sr=5,
    seed=9506,
):
    """
    Génère :
      - 1 rayon principal par pixel
      - nu_sr * nv_sr sous-rayons jitter stratifiés
    """
    rays_dirs = []
    rays_weights = []
    rays_pixel_ids = []
    rays_is_main = []
    nu_pix = len(u_edges) - 1
    nv_pix = len(v_edges) - 1
    G_r = np.zeros(nu_pix * nv_pix, dtype=np.float64)
    R = tel.rotation_matrix
    rng = np.random.default_rng(seed)
    for j in range(nv_pix):
        for i in range(nu_pix):
            u0, u1 = u_edges[i], u_edges[i+1]
            v0, v1 = v_edges[j], v_edges[j+1]
            du = (u1 - u0) / nu_sr
            dv = (v1 - v0) / nv_sr
            pixel_id = i * nv_pix + j
            # ==========================================================
            # 1) Rayon principal (géométriquement défini)
            # ==========================================================
            u_center = 0.5 * (u0 + u1)
            v_center = 0.5 * (v0 + v1)
            dir_local = np.array(
                [-u_center, -v_center, 1.0],
                dtype=float,
            )
            S = max(0.0, Lx - abs(u_center) * delta_z) \
              * max(0.0, Ly - abs(v_center) * delta_z)
            dG = S * (u1 - u0) * (v1 - v0) / (1 + u_center**2 + v_center**2)**2
            if dG > 0.0:
                dir_global = R @ dir_local
                rays_dirs.append(dir_global)
                rays_weights.append(dG)
                rays_pixel_ids.append(pixel_id)
                rays_is_main.append(True)
                G_r[pixel_id] += dG
            # ==========================================================
            # 2) Sous-rayons jitter stratifiés
            # ==========================================================
            for iu in range(nu_sr):
                for iv in range(nv_sr):
                    u = u0 + (iu + rng.random()) * du
                    v = v0 + (iv + rng.random()) * dv
                    dir_local = np.array(
                        [-u, -v, 1.0],
                        dtype=float,
                    )
                    S = max(0.0, Lx - abs(u) * delta_z) \
                      * max(0.0, Ly - abs(v) * delta_z)
                    dG = S * du * dv / (1 + u**2 + v**2)**2
                    if dG <= 0.0:
                        continue
                    dir_global = R @ dir_local
                    rays_dirs.append(dir_global)
                    rays_weights.append(dG)
                    rays_pixel_ids.append(pixel_id)
                    rays_is_main.append(False)
                    G_r[pixel_id] += dG
    return (
        np.asarray(rays_dirs, dtype=np.float64),
        np.asarray(rays_weights, dtype=np.float64),
        np.asarray(rays_pixel_ids, dtype=np.int64),
        np.asarray(rays_is_main, dtype=bool),
        G_r,
    )

@numba.njit(inline="always")
def voxel_id_vtk(i, j, k, nx, ny, nz):
    return i + nx * (j + ny * k) #F-order
    # return k + nz * (j + ny * i) #C-order

@numba.njit
def ray_aabb_intersection(x0, y0, z0, dx, dy, dz,
                          xmin, xmax, ymin, ymax, zmin, zmax):
    '''
    Test d'intersection d'un rayon avec une boîte englobante axis-aligned (AABB).
    Retourne un booléen indiquant s'il y a intersection, et les distances tmin et tmax le long du rayon où il entre et sort de la boîte.
    Paramètres
    ----------
    (x0, y0, z0) : origine du rayon
    (dx, dy, dz) : direction du rayon 
    (xmin, xmax, ymin, ymax, zmin, zmax) : limites de la boîte englobante
    '''
    tmin = -1e30
    tmax =  1e30
    for o, d, lo, hi in (
        (x0, dx, xmin, xmax),
        (y0, dy, ymin, ymax),
        (z0, dz, zmin, zmax),
    ):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return False, 0.0, 0.0
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False, 0.0, 0.0
    return True, tmin, tmax


@numba.njit
def trace_ray(
    x0, y0, z0,
    dx, dy, dz,
    x_edges, y_edges, z_edges,
    nx, ny, nz,
    mask_voxel,
    rows, cols, data, counts,
    pixel_id, weight,
    is_main_ray=False,
    rays_length=None,
    max_dist=1.2e3,
    eps=1e-12,
    # seed=9506
):
    '''
    Trace un rayon dans la grille voxel et accumule les longueurs d'intersection pondérées dans les listes rows, cols, data.
   
    Paramètres
    ----------
    (x0, y0, z0) : origine du rayon
    (dx, dy, dz) : direction du rayon        
    x_edges, y_edges, z_edges : arrays de taille (N+1) donnant les bords des voxels
    nx, ny, nz : nombre de voxels dans chaque dimension
    mask_voxel : array 1D de taille (nx*ny*nz) indiquant quels voxels sont actifs (1) ou non (0)
    rows, cols, data : listes numba.typed.List pour accumuler les indices de voxels (rows), les indices de pixels (cols) et les longueurs d'intersection pondérées (data)
    pixel_id : indice du pixel correspondant à ce rayon
    weight : poids à appliquer à la longueur d'intersection (ex: acceptance)
    max_dist : distance maximale à parcourir le long du rayon
    '''

    #Axis-Aligned Bounding Box (AABB) intersection test to find entry and exit points of the ray in the grid
    hit, t0, t1 = ray_aabb_intersection(
        x0, y0, z0, dx, dy, dz,
        x_edges[0], x_edges[-1],
        y_edges[0], y_edges[-1],
        z_edges[0], z_edges[-1],
    )  
    if not hit or t1 <= 0:
        # print("hit:", hit, "t0:", t0, "t1:", t1)
        return
    # Point d'entrée dans la grille
    t = max(t0, 0.0)
    x = x0 + t * dx
    y = y0 + t * dy
    z = z0 + t * dz
    # amplitude = petite fraction de la taille voxel

    # indices initiaux
    i = np.searchsorted(x_edges, x, side="right") - 1
    j = np.searchsorted(y_edges, y, side="right") - 1
    k = np.searchsorted(z_edges, z, side="right") - 1
    
    # Vérification des indices initiaux (peuvent être hors limites si le point d'entrée est exactement sur une face)
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        # print(f"Initial indices out of bounds: i={i}, j={j}, k={k}")
        return
    # calcul des t pour les prochaines intersections avec les plans de la grille
    if dx == 0.0:
        tx = 1e30
        dtx = 1e30
        sx = 0
    else:
        sx = 1 if dx > 0 else -1
        xb = x_edges[i+1] if dx > 0 else x_edges[i]
        tx = t + (xb - x) / dx
        dtx = abs((x_edges[i+1] - x_edges[i]) / dx)
    if dy == 0.0:
        ty = 1e30
        dty = 1e30
        sy = 0
    else:
        sy = 1 if dy > 0 else -1
        yb = y_edges[j+1] if dy > 0 else y_edges[j]
        ty = t + (yb - y) / dy
        dty = abs((y_edges[j+1] - y_edges[j]) / dy)
    if dz == 0.0:
        tz = 1e30
        dtz = 1e30
        sz = 0
    else:
        sz = 1 if dz > 0 else -1
        zb = z_edges[k+1] if dz > 0 else z_edges[k]
        tz = t + (zb - z) / dz
        dtz = abs((z_edges[k+1] - z_edges[k]) / dz)
    # On avance dans la grille en choisissant à chaque étape le plan de grille le plus proche
    while 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and t < min(t1, max_dist):
        i0, j0, k0 = i, j, k
        # Calcul du prochain t d'intersection avec un plan de grille
        if tx <= ty and tx <= tz:
        # if tx < ty - eps and tx < tz - eps:
            t_next = tx
            tx += dtx
            i += sx
        elif ty <= tz:
            t_next = ty
            ty += dty
            j += sy
        else:
            t_next = tz
            tz += dtz
            k += sz
        length = max(0.0, min(t_next, t1) - t)
        t = t_next
        vid = voxel_id_vtk(i0, j0, k0, nx, ny, nz)
        if mask_voxel[vid]:
            rows.append(vid)
            cols.append(pixel_id)
            data.append(length*weight)
            counts.append(1)
            if is_main_ray and rays_length is not None:
                rays_length[pixel_id] += length

@numba.njit
def trace_ray_subvox(
    x0, y0, z0,
    dx, dy, dz,
    x_edges, y_edges, z_edges,
    nx, ny, nz,
    mask_voxel,
    rows, cols, data,
    pixel_id, weight,
    is_main_ray=False,
    rays_length=None,
    max_dist=1.2e3,
    Ns=1   # nombre de sous-segments par voxel traversé
):

    hit, t0, t1 = ray_aabb_intersection(
        x0, y0, z0, dx, dy, dz,
        x_edges[0], x_edges[-1],
        y_edges[0], y_edges[-1],
        z_edges[0], z_edges[-1],
    )

    if not hit or t1 <= 0.0:
        return

    t = max(t0, 0.0)

    x = x0 + t * dx
    y = y0 + t * dy
    z = z0 + t * dz

    i = np.searchsorted(x_edges, x, side="right") - 1
    j = np.searchsorted(y_edges, y, side="right") - 1
    k = np.searchsorted(z_edges, z, side="right") - 1

    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return

    # Préparation DDA
    if dx == 0.0:
        tx = 1e30; dtx = 1e30; sx = 0
    else:
        sx = 1 if dx > 0 else -1
        xb = x_edges[i+1] if dx > 0 else x_edges[i]
        tx = t + (xb - x) / dx
        dtx = abs((x_edges[i+1] - x_edges[i]) / dx)

    if dy == 0.0:
        ty = 1e30; dty = 1e30; sy = 0
    else:
        sy = 1 if dy > 0 else -1
        yb = y_edges[j+1] if dy > 0 else y_edges[j]
        ty = t + (yb - y) / dy
        dty = abs((y_edges[j+1] - y_edges[j]) / dy)

    if dz == 0.0:
        tz = 1e30; dtz = 1e30; sz = 0
    else:
        sz = 1 if dz > 0 else -1
        zb = z_edges[k+1] if dz > 0 else z_edges[k]
        tz = t + (zb - z) / dz
        dtz = abs((z_edges[k+1] - z_edges[k]) / dz)

    eps = 1e-12

    while (0 <= i < nx and
           0 <= j < ny and
           0 <= k < nz and
           t < min(t1, max_dist)):

        i0, j0, k0 = i, j, k

        # prochain plan
        t_next = tx
        if ty < t_next:
            t_next = ty
        if tz < t_next:
            t_next = tz

        seg_length = max(0.0, min(t_next, t1) - t)

        if seg_length > 0.0:
            # subdivision interne
            dl = seg_length / Ns
            for s in range(Ns):
                t_mid = t + (s + 0.5) * dl
                x_mid = x0 + t_mid * dx
                y_mid = y0 + t_mid * dy
                z_mid = z0 + t_mid * dz
                ii = np.searchsorted(x_edges, x_mid, side="right") - 1
                jj = np.searchsorted(y_edges, y_mid, side="right") - 1
                kk = np.searchsorted(z_edges, z_mid, side="right") - 1
                if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                    vid = voxel_id_vtk(ii, jj, kk, nx, ny, nz)
                    if mask_voxel[vid]:
                        rows.append(vid)
                        cols.append(pixel_id)
                        data.append(weight * dl)
                        if is_main_ray and rays_length is not None:
                            rays_length[pixel_id] += dl
        t = t_next
        # Avance multi-axes propre
        if abs(tx - t_next) < eps:
            tx += dtx
            i += sx

        if abs(ty - t_next) < eps:
            ty += dty
            j += sy

        if abs(tz - t_next) < eps:
            tz += dtz
            k += sz


@numba.njit
def accumulate_matrix(
    rays_dirs,
    rays_weights,
    rays_pixel_ids,
    rays_is_main,
    x0, y0, z0,
    mask_voxel,
    x_edges, y_edges, z_edges,
    rays_length,  # array pour stocker la longueur totale d'intersection de chaque rayon principal
    max_dist, 
    eps=1e-12, 
    # seed=9506
):
    '''    
    Accumule les longueurs d'intersection pondérées de tous les rayons dans une matrice creuse au format COO (rows, cols, data).
    
    Paramètres
    ----------
    u_edges, v_edges : arrays de taille (N+1) donnant les bords angulaires des pixels (tan(theta_x), tan(theta_y))
    rays_dirs : (Nrays, 3) directions des rayons
    rays_weights : (Nrays,) poids à appliquer à chaque rayon (ex: acceptance)
    rays_pixel_ids : (Nrays,) indice du pixel correspondant à chaque rayon     
    (x0, y0, z0) : origine des rayons   
    mask_voxel : array 1D de taille (nx*ny*nz) indiquant quels voxels sont actifs (1) ou non (0)
    x_edges, y_edges, z_edges : arrays de taille (N+1) donnant les bords des voxels
    max_dist : distance maximale à parcourir le long du rayon
    '''
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1
    rows = numba.typed.List.empty_list(numba.int64)
    cols = numba.typed.List.empty_list(numba.int64)
    data = numba.typed.List.empty_list(numba.float64)
    counts = numba.typed.List.empty_list(numba.float64)
    for r in range(rays_dirs.shape[0]):
        dx, dy, dz = rays_dirs[r]
        w = rays_weights[r]
        pixel_id = rays_pixel_ids[r]
        is_main_ray = rays_is_main[r]
        trace_ray(
            x0, y0, z0,
            dx, dy, dz,
            x_edges, y_edges, z_edges,
            nx, ny, nz,
            mask_voxel,
            rows, cols, data, counts,
            pixel_id=pixel_id, 
            weight = w,
            is_main_ray=is_main_ray,
            rays_length=rays_length,
            max_dist=max_dist,
            eps=eps,
            # seed=seed
        )
    return rows, cols, data, counts


def generate_rays(tel, conf, nsr=5):
    size_panel = conf.panels[0].matrix.scintillator.length
    delta_z = conf.length_z
    return set_rays_with_acceptance_midpoint(
        tel,
        u_edges=conf.u_edges,
        v_edges=conf.v_edges,
        delta_z=delta_z * 1e-3,
        Lx=size_panel * 1e-3,
        Ly=size_panel * 1e-3,
        n_sr=nsr,
    )

def filter_rays(subrays_dirs, subrays_weights, subrays_pixel_ids, subrays_is_main):
    mask = (subrays_dirs[:,2] > 0)
    return (
        subrays_dirs[mask],
        subrays_weights[mask],
        subrays_pixel_ids[mask],
        subrays_is_main[mask],
    )

def compute_voxel_ray_matrix(
    rays_dirs,
    rays_weights,
    rays_pixel_ids,
    rays_is_main,
    geom,
    origin,
    max_dist,
    rays_length
):
    rows, cols, data, counts = accumulate_matrix(
        rays_dirs=rays_dirs,
        rays_weights=rays_weights,
        rays_pixel_ids=rays_pixel_ids,
        rays_is_main=rays_is_main,
        x0=origin[0],
        y0=origin[1],
        z0=origin[2],
        mask_voxel=geom.mask_voxel,
        x_edges=geom.x_edges,
        y_edges=geom.y_edges,
        z_edges=geom.z_edges,
        max_dist=max_dist,
        rays_length=rays_length,
    )
    n_voxels = (
        (len(geom.x_edges)-1) *
        (len(geom.y_edges)-1) *
        (len(geom.z_edges)-1)
    )
    n_pixels = len(rays_length)
    M = sp.coo_matrix(
        (np.array(data),(np.array(rows),np.array(cols))),
        shape=(n_voxels,n_pixels)
    ).tocsr()
    return M



def process_configuration(h5file, tel, conf, geom, dirs,  vtkfile=None):
    subrays_dirs, subrays_weights, subrays_pixel_ids, subrays_is_main, G_r = \
        generate_rays(tel, conf)
    n_pixels = np.count_nonzero(subrays_is_main)
    subrays_dirs, subrays_weights, subrays_pixel_ids, subrays_is_main = \
        filter_rays(
            subrays_dirs,
            subrays_weights,
            subrays_pixel_ids,
            subrays_is_main
        )
    rays_length = np.zeros(n_pixels, dtype=np.float64)
    origin = tel.coordinates
    M = compute_voxel_ray_matrix(
        subrays_dirs,
        subrays_weights,
        subrays_pixel_ids,
        subrays_is_main,
        geom,
        origin,
        max_dist=1.2e3,
        rays_length=rays_length
    )
    # fout = dirs["voxel"] / f"voxel_ray_matrix_{tel.name}_{conf_name}.npz"
    if h5file is not None:
        save_sparse_matrix_hdf5(
                    h5file,
                    tel.name,
                    conf.name,
                    M, 
                    acceptance=G_r,
                    rays_length=rays_length,
                )
    if vtkfile is not None:
        export_voxel_intersections_vtk_fast(
            vtkfile=vtkfile,
            M=M,
            voxel_volume=geom.voxel_volume,
            voxel_volume_vtk=geom.voxel_volume_vtk,
            rays_length=rays_length,
            # rays_theta=rays_theta,
            tel_name=tel.name,
            conf_name=conf.name,
            vs=vs,
            dir_voxel=dirs["voxel"],
        )
    # np.savez_compressed(
    #     fout,
    #     data=M.data,
    #     indices=M.indices,
    #     indptr=M.indptr,
    #     shape=M.shape,
    #     acceptance=G_r,
    #     rays_length=rays_length,
    #     x_edges=geom.x_edges,
    #     y_edges=geom.y_edges,
    #     z_edges=geom.z_edges,
    # )
    # matrices[tel.name][conf_name] = {
    #     "data": M.data,
    #     "indices": M.indices,
    #     "indptr": M.indptr,
    #     "shape": M.shape, 
    #     "acceptance":G_r,
    #     "rays_length":rays_length,
    # }
    

def process_telescope(h5file, tel, geom, dirs, vtkfile):
    tel.compute_angular_coordinates()
    tel.adjust_height(
        geom.x_edges,
        geom.y_edges,
        geom.z_edges,
        geom.mask_voxel
    )
    for _, conf in tel.configurations.items():
        process_configuration(
            h5file,
            tel,
            conf,
            geom,
            dirs, 
            vtkfile
        )

# def save_sparse_matrix_hdf5(h5file, tel_name, conf_name, M, acceptance, rays_length):
#     grp_tel = h5file.require_group(tel_name)
#     grp = grp_tel.require_group(conf_name)
#     grp.create_dataset("data", data=M.data, compression="gzip")
#     grp.create_dataset("indices", data=M.indices, compression="gzip")
#     grp.create_dataset("indptr", data=M.indptr, compression="gzip")
#     grp.create_dataset("shape", data=M.shape)
#     grp.create_dataset("acceptance", data=acceptance)
#     grp.create_dataset("rays_length", data=rays_length)

def save_sparse_matrix_hdf5(h5file, tel_name, conf_name, M, acceptance, rays_length):
    grp_tel = h5file.require_group(tel_name)
    grp = grp_tel.require_group(conf_name)
    def overwrite_dataset(group, name, data, **kwargs):
        if name in group:
            del group[name]  # supprime l'ancien dataset
        group.create_dataset(name, data=data, compression="gzip", **kwargs)
    overwrite_dataset(grp, "data", M.data)
    overwrite_dataset(grp, "indices", M.indices)
    overwrite_dataset(grp, "indptr", M.indptr)
    if "shape" in grp:
        del grp["shape"]
    grp.create_dataset("shape", data=M.shape)
    overwrite_dataset(grp, "acceptance", acceptance)
    overwrite_dataset(grp, "rays_length", rays_length)


def load_sparse_matrix_hdf5(file, tel_name, conf_name):
    grp = file[tel_name][conf_name]
    data = grp["data"][:]
    indices = grp["indices"][:]
    indptr = grp["indptr"][:]
    shape = tuple(grp["shape"][:])
    return sp.csr_matrix((data, indices, indptr), shape=shape)

def export_voxel_intersections_vtk_fast(
    vtkfile,
    M,
    voxel_volume,
    voxel_volume_vtk,
    rays_length,
    # rays_theta,
    tel_name,
    conf_name,
    vs,
    dir_voxel,
):

    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(vtkfile))
    reader.Update()
    grid_out = reader.GetOutput()

    grid_out.GetCellData().AddArray(voxel_volume_vtk)

    M = M.tocsr()

    indptr = M.indptr
    indices = M.indices
    data = M.data

    n_cells = M.shape[0]

    primary_pixel = vtk.vtkIntArray()
    primary_pixel.SetName("id_rays")
    primary_pixel.SetNumberOfTuples(n_cells)

    intercepted_weight = vtk.vtkFloatArray()
    intercepted_weight.SetName("intercepted_weight")
    intercepted_weight.SetNumberOfTuples(n_cells)

    intercepted_volume = vtk.vtkFloatArray()
    intercepted_volume.SetName("intercepted_volume")
    intercepted_volume.SetNumberOfTuples(n_cells)

    intercepted_length = vtk.vtkFloatArray()
    intercepted_length.SetName("intercepted_length")
    intercepted_length.SetNumberOfTuples(n_cells)

    n_intercepting_rays = vtk.vtkIntArray()
    n_intercepting_rays.SetName("num_intercepting_rays")
    n_intercepting_rays.SetNumberOfTuples(n_cells)
    # theta_rays_vtk = vtk.vtkFloatArray()
    # theta_rays_vtk.SetName("theta_rays")
    # theta_rays_vtk.SetNumberOfTuples(n_cells)
    for vid in tqdm(range(n_cells), desc="To VTK"):
        start = indptr[vid]
        end = indptr[vid + 1]
        if start != end:
            cols = indices[start:end]
            weights = data[start:end]
            local_max = np.argmax(weights)
            pid = int(cols[local_max])
            w = float(weights[local_max])
            primary_pixel.SetValue(vid, pid)
            intercepted_weight.SetValue(vid, w)
            intercepted_length.SetValue(vid, float(rays_length[pid]))
            intercepted_volume.SetValue(vid, float(voxel_volume[vid]))
            n_intercepting_rays.SetValue(vid, end - start)
            # theta_rays_vtk.SetValue(vid, float(rays_theta[pid]))
        else:

            primary_pixel.SetValue(vid, -1)
            intercepted_weight.SetValue(vid, 0.0)
            intercepted_length.SetValue(vid, 0.0)
            intercepted_volume.SetValue(vid, 0.0)
            n_intercepting_rays.SetValue(vid, 0)
            # theta_rays_vtk.SetValue(vid, 0.0)

    grid_out.GetCellData().AddArray(primary_pixel)
    grid_out.GetCellData().AddArray(intercepted_weight)
    grid_out.GetCellData().AddArray(intercepted_volume)
    grid_out.GetCellData().AddArray(intercepted_length)
    grid_out.GetCellData().AddArray(n_intercepting_rays)
    # grid_out.GetCellData().AddArray(theta_rays_vtk)

    f_vts = dir_voxel / f"voxel_intercepted_{tel_name}_{conf_name}_vox{vs}m.vts"

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(f_vts))
    writer.SetInputData(grid_out)
    writer.Write()


if __name__ == "__main__":
    
    survey = CURRENT_SURVEY
    struct_dir = STRUCT_DIR / survey.name
    dirs = {
        "survey": struct_dir,
        "voxel": struct_dir/"voxel",
        "png": struct_dir / "png",
        "tel": struct_dir/ "telescope",
    }
    vs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    vtkfile = dirs["voxel"] / f"topo_voi_vox{vs}m.vts"
    grid, geom = load_voxel_grid(vtkfile)
    matrices= {}
    basename = "real_telescopes"
    dtel = survey.telescopes 
    dtel = {"SBR": dtel["SBR"], "SXF": dtel["SXF"]} # pour test rapide
    # basename = f"toy_telescopes_s9506"   
    # fin_toytel = dirs["tel"] / f"{basename}_vox{vs}m.pkl"
    # with open(fin_toytel, 'rb') as f:
    #     dtel = pickle.load(f) 

    h5_path = dirs["voxel"] / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"
    with h5py.File(h5_path, "r+") as file:
        for i, tel in tqdm(enumerate(dtel.values()), total=len(dtel), desc="Progress"):
                process_telescope(
                    file,
                    tel,
                    geom,
                    dirs, 
                    None,
                )
    print(f"Saved {h5_path}")

    ###Test read h5file
    # with h5py.File(h5_path) as f:
    #     telescopes = list(f.keys())
    #     print(telescopes)   
    #     M = load_sparse_matrix_hdf5(f, telescopes[0], "3p1")
    #     print(M.shape)