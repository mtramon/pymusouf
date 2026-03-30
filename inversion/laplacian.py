#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from tqdm import tqdm

def build_laplacian_3d(nx, ny, nz):
    """
    Laplacien 3D 6-voisins sur grille régulière.
    Retourne une matrice CSR (nvox, nvox).
    """
    nvox = nx * ny * nz
    rows = []
    cols = []
    data = []
    def vid(i, j, k):
        return i + nx * (j + ny * k) #F-like
        # return k + nz * (j + ny * i) #C-like

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = vid(i, j, k)
                diag = 0
                # voisins 6-connectivité
                for di, dj, dk in [
                    (1,0,0), (-1,0,0),
                    (0,1,0), (0,-1,0),
                    (0,0,1), (0,0,-1)
                ]:
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor = vid(ni, nj, nk)
                        rows.append(center)
                        cols.append(neighbor)
                        data.append(1.0)
                        diag -= 1
                # diagonale
                rows.append(center)
                cols.append(center)
                data.append(-diag)
    L = sp.coo_matrix((data, (rows, cols)), shape=(nvox, nvox))
    return L.tocsr()


def build_exp_distance_regularization(
    coords: np.ndarray,
    length: float,
    sigma: float,
    r_cut: float = None,
):
    """
    Construit une matrice de régularisation exponentielle tronquée :
        C_ij = sigma^2 * exp(-d_ij / length)
    Retourne directement C^{-1} approx creuse.
    """
    nvox = coords.shape[0]

    if r_cut is None:
        r_cut = 3.0 * length
    tree = cKDTree(coords)
    rows = []
    cols = []
    data = []
    for i in tqdm(range(nvox), desc="Cov"):
        neighbors = tree.query_ball_point(coords[i], r_cut)
        for j in neighbors:
            d = np.linalg.norm(coords[i] - coords[j])
            val = sigma**2 * np.exp(-(d / length))
            rows.append(i)
            cols.append(j)
            data.append(val)
    C = sp.coo_matrix((data, (rows, cols)), shape=(nvox, nvox)).tocsr()
    # Approximation : inversion locale par solveur creux
    # (acceptable car matrice creuse et SPD)
    C_inv = sp.linalg.inv(C)
    return C_inv.tocsr()

def build_graph_laplacian(coords, length, r_cut=None, kernel='exponential'):
    """
    Construit un Laplacien de graphe à partir d'un noyau exponentiel.
    
    Paramètres
    ----------
    coords : ndarray, shape (n_voxels, 3)
        Coordonnées des centres des voxels.
    length : float
        Longueur de corrélation (échelle du lissage).
    r_cut : float, optionnel
        Rayon de coupure pour la recherche de voisins. Par défaut 3*length.
    
    Retourne
    --------
    L : csr_matrix, shape (n_voxels, n_voxels)
        Laplacien de graphe (symétrique, semi-défini positif).
    """
    nvox = coords.shape[0]
    if r_cut is None:
        r_cut = 3.0 * length
    tree = cKDTree(coords)
    rows, cols, data = [], [], []
    # Parcours de tous les voxels
    for i in tqdm(range(nvox), desc="Construction du graphe"):
        voisins = tree.query_ball_point(coords[i], r_cut)
        for j in voisins:
            if j > i:  # éviter les doublons (on remplira symétriquement)
                d = np.linalg.norm(coords[i] - coords[j])
                if kernel == 'exponential':
                    w = np.exp(-d / length)
                elif kernel == 'gaussian':
                    w = np.exp(-(d / length)**2)
                else:
                    raise ValueError("kernel must be 'exponential' or 'gaussian'")
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([w, w])
    # Matrice d'adjacence W
    W = sp.coo_matrix((data, (rows, cols)), shape=(nvox, nvox)).tocsr()
    # Degrés
    deg = np.array(W.sum(axis=1)).ravel()
    # Laplacien : D - W
    L = sp.diags(deg) - W
    return L.tocsr()