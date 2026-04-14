#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de modèles synthétiques de densité pour un volcan (grille structurée VTK).
Chaque modèle contient de 1 à 3 structures géologiques plausibles :
- dykes verticaux
- cavités (zones de faible densité)
- zones d'altération hydrothermale (densité réduite)
- cheminées / panaches remontant vers la surface
Les modèles sont sauvegardés au format VTK pour visualisation dans ParaView.
"""

import json
import numpy as np
import numba
import random
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree, KDTree
import sys 
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from inversion.voxelgrid import load_voxel_grid, get_coordinates
from inversion.topography import build_interpolator_topography, interpolate_topography
from utils.tools import check_array_order

# ----------------------------------------------------------------------
# Paramètres modifiables
# ----------------------------------------------------------------------
N_MODELS = 500                               # Nombre de modèles à générer
SEED = 9506                                   # Pour reproductibilité
BACKGROUND_DENSITY = 2.3                    # g/cm³, densité de fond
BACKGROUND_DENSITY_RANGE = (2.3, 2.4)                    # g/cm³, densité de fond

# Densités des anomalies (g/cm³)
DIKE_DENSITY = 2.7
CAVITY_DENSITY = 1.2                       
ALTERED_DENSITY = 1.8
PLUME_DENSITY = 2.0

NON_ALTERED_DENSITY_RANGE = (2.6,2.7)
DIKE_DENSITY_RANGE = (2.6,2.7)
CAVITY_DENSITY_RANGE = (1.7, 1.9)                     
ALTERED_DENSITY_RANGE = (1.6, 2.3)
PLUME_DENSITY_RANGE = (1.8, 2.0)

# Plages pour les paramètres aléatoires
NON_ALTERED_RADIUS_RANGE = (50,300)
NON_ALTERED_HEIGHT_RANGE = (50,150)
PLUME_RADIUS_RANGE = (50, 200)                # m
PLUME_HEIGHT_RANGE = (200, 300)              # m

DIKE_THICKNESS_RANGE = (5, 20)               # m
DIKE_LENGTH_RANGE = (100, 300)               # m
CAVITY_RADIUS_RANGE = (10, 50)               # m
ALTERED_RADIUS_RANGE = (30, 100)          # m

# Région centrale (distance horizontale max depuis le centre)
RADIUS_CENTRAL = 350.0                        # m

# ----------------------------------------------------------------------
# Fonctions de génération des structures
# ----------------------------------------------------------------------
def generate_dike(density_field, x, y, z, mask, density_value,
                  center, strike, dip, thickness, length):
    """
    Ajoute un dyke planaire.
    Paramètres :
        density_field : tableau 3D (nx, ny, nz) modifiable
        x, y, z : coordonnées des centres des voxels (grilles 3D)
        mask : masque 3D des voxels du volcan (bool)
        density_value : densité du dyke
        center : tuple (xc, yc, zc) point par lequel passe le plan du dyke
        strike : angle en degrés (0 = axe Y, 90 = axe X) – direction horizontale
        dip : angle en degrés (0 = vertical, 90 = horizontal)
        thickness : épaisseur (m)
        length : longueur (m) – étendue le long de la direction
    """
    # Normaliser les directions
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)

    # Vecteur normal au plan du dyke (perpendiculaire)
    nx = np.sin(dip_rad) * np.cos(strike_rad)
    ny = np.sin(dip_rad) * np.sin(strike_rad)
    nz = np.cos(dip_rad)

    # Vecteur direction de la longueur (dans le plan, horizontal)
    if abs(np.cos(strike_rad)) > 0.5:
        # direction perpendiculaire à la normale (approximative)
        ux = -np.sin(strike_rad)
        uy = np.cos(strike_rad)
        uz = 0.0
    else:
        ux = 0.0
        uy = 0.0
        uz = 1.0  # vertical

    # Parcourir les voxels
    for i in range(density_field.shape[0]):
        for j in range(density_field.shape[1]):
            for k in range(density_field.shape[2]):
                if not mask[i, j, k]:
                    continue
                # Coordonnées du voxel
                xc = x[i, j, k]
                yc = y[i, j, k]
                zc = z[i, j, k]
                # Vecteur depuis le centre
                dx = xc - center[0]
                dy = yc - center[1]
                dz = zc - center[2]
                # Distance au plan
                dist_plane = abs(dx * nx + dy * ny + dz * nz)
                # Distance le long de la direction de longueur
                proj_len = abs(dx * ux + dy * uy + dz * uz)
                if dist_plane <= thickness/2 and proj_len <= length/2:
                    density_field[i, j, k] = random.uniform(*density_value)

def generate_cavity(density_field, x, y, z, mask, density_value,
                    center, radius):
    """
    Ajoute une cavité sphérique.
    """
    for i in range(density_field.shape[0]):
        for j in range(density_field.shape[1]):
            for k in range(density_field.shape[2]):
                if not mask[i, j, k]:
                    continue
                xc = x[i, j, k]
                yc = y[i, j, k]
                zc = z[i, j, k]
                dist = np.sqrt((xc - center[0])**2 +
                               (yc - center[1])**2 +
                               (zc - center[2])**2)
                if dist <= radius:
                    density_field[i, j, k] = random.uniform(*density_value)

def generate_altered_zone(density_field, x, y, z, mask, density_value,
                             center, radii):
    """
    Zone ellipsoïdale de densité réduite (altération).
    radii = (rx, ry, rz)
    """
    rx, ry, rz = radii
    for i in range(density_field.shape[0]):
        for j in range(density_field.shape[1]):
            for k in range(density_field.shape[2]):
                if not mask[i, j, k]:
                    continue
                xc = x[i, j, k]
                yc = y[i, j, k]
                zc = z[i, j, k]
                # distance normalisée
                dx = (xc - center[0]) / rx
                dy = (yc - center[1]) / ry
                dz = (zc - center[2]) / rz
                if dx*dx + dy*dy + dz*dz <= 1.0:
                    density_field[i, j, k] = random.uniform(*density_value)


import numpy as np

def generate_ellipsoid_gradient(density_field, x, y, z, mask,
                                center, radii, orientation_matrix,
                                center_density, background_density,
                                profile='linear', power=2):
    """
    Génère une région ellipsoïdale avec un gradient de densité.
    La densité varie de center_density (au centre) à background_density (à la surface)
    selon le profil spécifié.

    Paramètres
    ----------
    density_field : ndarray 3D modifiable
    x, y, z : ndarrays 3D des coordonnées des centres des voxels
    mask : ndarray 3D booléen (True = voxel du volcan)
    center : tuple (xc, yc, zc) centre de l'ellipsoïde
    radii : tuple (rx, ry, rz) rayons selon les axes principaux
    orientation_matrix : ndarray (3,3) matrice de rotation des axes (ou None pour axes alignés)
    center_density : densité au centre
    background_density : densité à l'extérieur
    profile : 'constant', 'linear', 'quadratic', 'gaussian'
    power : exposant pour 'quadratic' (d^power) ou facteur d'échelle pour 'gaussian'
    """
    if orientation_matrix is None:
        orientation_matrix = np.eye(3)
    rx, ry, rz = radii
    nx, ny, nz = density_field.shape
    count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i, j, k]:
                    continue
                p = np.array([x[i, j, k], y[i, j, k], z[i, j, k]]) - np.array(center)
                # Rotation inverse pour se placer dans le repère de l'ellipsoïde
                p_rot = orientation_matrix.T @ p
                # Distance normalisée
                d2 = (p_rot[0]/rx)**2 + (p_rot[1]/ry)**2 + (p_rot[2]/rz)**2
                if d2 <= 1.0:
                    d = np.sqrt(d2)
                    # td = random.uniform(0, 0.5)
                    if d >= 0.5: 
                        if profile == 'constant':
                            factor = 1.0
                        elif profile == 'linear':
                            factor = 1 - d
                        elif profile == 'quadratic':
                            factor = 1 - d**power
                        elif profile == 'gaussian':
                            factor = np.exp(- (d**2) )
                        else:
                            factor = 1 - d
                        # Interpolation
                        val = background_density + (center_density - background_density) * factor
                    else : 
                        val = random.uniform(center_density-0.05, center_density+0.05)
                    density_field[i, j, k] = val
                    count += 1
                        
    return count


def generate_non_altered_region(density_field, x, y, z, mask,
                                center, radii, orientation_matrix,
                                center_density, background_density,
                                profile='constant', power=2):
    """
    Génère une région de roche non altérée (haute densité).
    Par défaut, profile='constant' donne une région homogène.
    """
    return generate_ellipsoid_gradient(density_field, x, y, z, mask,
                                       center, radii, orientation_matrix,
                                       center_density, background_density,
                                       profile, power)


def generate_plume_ellipsoid(density_field, x, y, z, mask,
                             bottom, top, radius,
                             center_density, background_density,
                             profile='linear', power=2):
    """
    Génère une plume ellipsoïdale allongée entre bottom et top,
    avec un gradient de densité depuis le centre (faible) vers la surface.

    Paramètres
    ----------
    density_field, x, y, z, mask : idem
    bottom, top : tuples (x,y,z) définissant l'axe principal
    radius : rayon transversal (m)
    center_density : densité au centre (faible)
    background_density : densité de fond
    profile : 'constant', 'linear', 'quadratic', 'gaussian'
    power : paramètre pour les profils non linéaires
    """
    p1 = np.array(bottom)
    p2 = np.array(top)
    # center=p1
    center = (p1 + p2) / 2.0
    direction = p2 - p1
    L = np.linalg.norm(direction)
    if L < 1e-6:
        return 0
    direction = direction / L

    # Construction d'une base orthonormée (u, v, w) où w = direction
    w = direction
    print(f"plume direction: {w}")
    # Trouver un vecteur non colinéaire
    t = np.array([1, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(w, t)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    # Matrice de rotation pour passer du repère global au repère local (u,v,w)
    # Les colonnes de orientation sont les vecteurs de la base locale exprimés dans le repère global
    orientation = np.column_stack((u, v, w))
    # print(f"plume orientation: {orientation}")

    # L'inverse est la transposée (car orthonormée)

    # Rayons : longitudinal = L/2, transversal = radius
    radii = (radius, radius, L/2.0)

    return generate_ellipsoid_gradient(density_field, x, y, z, mask,
                                       center, radii, orientation,
                                       center_density, background_density,
                                       profile, power)



# ----------------------------------------------------------------------
# Lecture du fichier VTK d'entrée
# ----------------------------------------------------------------------
def read_vtk_structured_grid(filename):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    grid = reader.GetOutput()
    return grid

def random_point_in_cylinder(center_xy, radius, z_range):
    """
    Tire un point aléatoire dans un cylindre vertical de rayon donné.
    Retourne (x, y, z).
    """
    # tirage uniforme dans le disque
    r = radius * np.sqrt(random.random())  # pour une densité surfacique uniforme
    theta = random.uniform(0, 2*np.pi)
    x = center_xy[0] + r * np.cos(theta)
    y = center_xy[1] + r * np.sin(theta)
    z = random.uniform(z_range[0], z_range[1])
    return (x, y, z)

def smooth_model_gaussian(density, sigma=1.0):
    """
    Applique un filtre gaussien 3D au modèle de densité.
    sigma : écart-type du noyau gaussien (en voxels).
    """
    return gaussian_filter(density, sigma=sigma, mode='constant', cval=0.0)

# ----------------------------------------------------------------------
# Sauvegarde d'un modèle en VTK
# ----------------------------------------------------------------------
def save_model_vtk(grid_template, density_3d, filename, array_name="density"):
    """
    Crée une copie de la grille d'entrée et y ajoute le champ de densité.
    """
    # Créer une copie profonde
    writer = vtk.vtkXMLStructuredGridWriter()
    grid = vtk.vtkStructuredGrid()
    grid.DeepCopy(grid_template)

    # Convertir le tableau numpy en vtkArray
    density_flat = density_3d.ravel(order='F')
    vtk_array = numpy_support.numpy_to_vtk(density_flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName(array_name)
    grid.GetCellData().AddArray(vtk_array)

    writer.SetFileName(str(filename))
    writer.SetInputData(grid)
    writer.Write()

def save_model_npz(density_3d, mask_3d, filename):
    """
    Sauvegarde le modèle au format NPZ (pour entraînement).
    """
    np.savez_compressed(filename,
                        density=density_3d.astype(np.float32),
                        mask=mask_3d.astype(bool))



def generate_winding_plume(density_field, x, y, z, mask,
                           bottom, depth,
                           interp,  # fonction interpolation surface(x,y) pour la hauteur maximale
                           R0=100, R1=30,       # rayons initial et final (m)
                           decay='linear',      # 'linear' ou 'exponential'
                           num_control_points=5,
                           smoothness=1.0,
                           center_density=1.8,
                           background_density=2.4,
                           profile='linear',
                           power=2,
                           seed=None):
    """
    Génère une plume tortueuse partant de z_min et remontant vers la surface.
    Le chemin est une courbe 3D aléatoire. Le rayon décroît de R0 à R1 le long de la courbe.
    """
    if seed is not None:
        np.random.seed(SEED)

    # Déterminer le point de départ (aléatoire dans le plan xy, dans la zone centrale)
    x0, y0, z_min = bottom
    
    # On peut restreindre pour rester dans le masque, mais on suppose que c'est bon

    # Générer des points de contrôle le long de la verticale avec des déviations horizontales
    t_vals = np.linspace(0, 1, num_control_points)
    z_surf0 = interpolate_topography(interp, x0, y0) 

    z_max = z_surf0 - depth

    # Déviations aléatoires en x et y (cumulatives pour donner une courbe)
    dx = np.cumsum(np.random.randn(num_control_points) * smoothness * (z_max - z_min) / num_control_points)
    dy = np.cumsum(np.random.randn(num_control_points) * smoothness * (z_max - z_min) / num_control_points)
    # Ajuster pour que le point final soit à la surface (éventuellement avec une condition de surface)

    z_max = interpolate_topography(interp, x0+dx[-1], y0+dy[-1]) 
    z_max -= depth 

    z_vals = z_min + (z_max - z_min) * t_vals
    
    x_vals = x0 + dx
    y_vals = y0 + dy

    # Points de contrôle pour la spline (on ajoute le point de départ et d'arrivée)
    pts = np.array([x_vals, y_vals, z_vals]).T

    # Interpolation spline (lisse)
    try:
        tck, u = splprep(pts.T, s=0)  # s=0 pour interpolation exacte
    except:
        # Si échec, utiliser une interpolation linéaire
        tck = None

    # Échantillonner la courbe en de nombreux points pour les calculs de distance
    num_samples = 200
    u_fine = np.linspace(0, 1, num_samples)
    if tck is not None:
        curve = splev(u_fine, tck)
        curve_pts = np.array(curve).T  # (num_samples, 3)
    else:
        # Interpolation linéaire entre les points de contrôle
        curve_pts = np.interp(u_fine[:, None], t_vals, pts)

    # Rayon le long de la courbe (fonction décroissante)
    if decay == 'linear':
        radii = R0 + (R1 - R0) * u_fine
    elif decay == 'exponential':
        # R0 * exp(-alpha * u) avec alpha tel que R1 = R0 * exp(-alpha)
        alpha = -np.log(R1/R0) if R1>0 else 1.0
        radii = R0 * np.exp(-alpha * u_fine)
    else:
        radii = np.linspace(R0, R1, num_samples)

    # Pour chaque voxel, trouver le point de la courbe le plus proche (distance minimale)
    # On peut utiliser un arbre k-d pour accélérer
    tree = cKDTree(curve_pts)

    nx, ny, nz = density_field.shape
    count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i, j, k]:
                    continue
                p = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                # Trouver l'indice du point de courbe le plus proche
                dist, idx = tree.query(p)
                # Si la distance est inférieure au rayon à ce point
                if dist <= radii[idx]:
                    # Calcul de la distance normalisée
                    d_norm = dist / radii[idx]
                    # Profil de densité
                    if profile == 'constant':
                        factor = 1.0
                    elif profile == 'linear':
                        factor = 1 - d_norm
                    elif profile == 'quadratic':
                        factor = 1 - d_norm**power
                    elif profile == 'gaussian':
                        factor = np.exp(- (d_norm**2) * power)
                    else:
                        factor = 1 - d_norm
                    val = background_density + (center_density - background_density) * factor
                    density_field[i, j, k] = val
                    count += 1
    return count


def generate_high_density_from_surface(density_field, x, y, z, mask,
                                       interp, density_high,
                                       background_density,
                                       depth_min, depth_max, seed=None, offset_z=100):
    """
    Génère une couche superficielle de haute densité (roche non altérée)
    depuis la surface jusqu'à une profondeur variable.
    
    Paramètres
    ----------
    density_field : ndarray 3D modifiable
    x, y, z : ndarrays 3D des coordonnées des centres des voxels
    mask : ndarray 3D booléen
    interp : fonction surface(x, y) -> z_surface
    density_high : densité de la roche non altérée (g/cm³)
    background_density : densité de fond
    depth_min, depth_max : plage de profondeur (m) sous la surface
    seed : pour reproductibilité
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Récupérer les coordonnées uniques en x,y pour générer une carte de profondeur aléatoire
    # Ici on utilise un bruit simple, mais on pourrait utiliser un champ de bruit plus sophistiqué
    nx, ny, nz = density_field.shape
    # On parcourt tous les voxels
    for i in range(nx):
        for j in range(ny):
            xi = x[i, j, 0]  # x constant pour cette colonne
            yj = y[i, j, 0]  # y constant
            z_surf = interpolate_topography(interp, xi, yj) + offset_z
            # Profondeur aléatoire pour cette colonne (fixe pour tous les k)
            # On utilise une graine dérivée de (i,j) pour reproductibilité
            rng = np.random.RandomState(seed + i*ny + j if seed else None)
            depth_col = rng.uniform(depth_min, depth_max) + offset_z
            z_bottom = z_surf - depth_col
            for k in range(nz):
                if not mask[i, j, k]:
                    continue
                zk = z[i, j, k]
                if zk <= z_surf and zk >= z_bottom:
                    # Interpolation linéaire en fonction de la profondeur (optionnel)
                    # Par exemple, décroissance linéaire vers le bas
                    t = (zk - z_bottom) / (z_surf - z_bottom)  # 0 au bas, 1 en surface
                    density_field[i, j, k] = background_density + (density_high - background_density) * t
    return


def generate_surface_high_density_region(density_field, x, y, z, mask,
                                         surface_point, depth,
                                         density_high, background_density,
                                         shape='ellipsoid',
                                         half_width=50, half_length=50,
                                         transition_width=20,
                                         vertical_gradient=True):
    """
    Génère une région de haute densité centrée en surface_point (x0,y0,z0)
    avec une profondeur 'depth' (m) sous la surface.
    La région a une forme horizontale définie par half_width et half_length
    (demi-dimensions selon x et y). La transition sur les bords est progressive
    sur 'transition_width' mètres. Un gradient vertical peut être appliqué
    (densité max en surface, décroissant vers le fond).
    
    shape : 'ellipsoid' ou 'rectangular'
    """
    x0, y0, z_surface = surface_point
    z_bottom = z_surface - depth
    
    nx, ny, nz = density_field.shape
    # Précalcul des indices de voxels dans la boîte englobante pour accélérer
    # (optionnel mais recommandé si le volume est grand)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i, j, k]:
                    continue
                # Coordonnées du voxel
                xc = x[i, j, k]
                yc = y[i, j, k]
                zc = z[i, j, k]
                # Vérifier si le voxel est dans la colonne de la région
                if zc > z_surface or zc < z_bottom:
                    continue
                # Distance horizontale normalisée
                if shape == 'ellipsoid':
                    dx = (xc - x0) / half_width
                    dy = (yc - y0) / half_length
                    d_horiz = np.sqrt(dx**2 + dy**2)
                else:  # rectangular
                    dx = abs(xc - x0) / half_width
                    dy = abs(yc - y0) / half_length
                    d_horiz = max(dx, dy)
                if d_horiz > 1.0:
                    continue
                # Distance horizontale au bord (0 au centre, 1 au bord)
                t_horiz = d_horiz
                # Facteur de transition latérale (1 au centre, 0 au bord)
                if t_horiz >= 1.0 - transition_width / half_width:
                    # Dans la zone de transition latérale
                    # On pourrait utiliser une interpolation linéaire
                    # Mais ici on applique un facteur qui décroît de 1 à 0
                    # sur l'épaisseur de transition
                    t_edge = (t_horiz - (1.0 - transition_width / half_width)) / (transition_width / half_width)
                    horiz_factor = 1.0 - t_edge
                else:
                    horiz_factor = 1.0
                # Facteur vertical (gradient)
                if vertical_gradient:
                    t_vert = (zc - z_bottom) / depth  # 0 au bas, 1 en surface
                    # On veut densité max en surface, min au bas
                    vert_factor = t_vert  # linéaire
                else:
                    vert_factor = 1.0
                
                # Densité finale
                density = background_density + (density_high - background_density) * horiz_factor * vert_factor
                density_field[i, j, k] = density

def generate_surface_feature_from_path(density_field, x, y, z, mask,
                                       z_surface, k_surface,
                                       center_xy, path_radius,
                                       depth_max, density_center, background_density,
                                       horizontal_decay='linear', vertical_decay='linear',
                                       n_points_path=10, seed=None):
    """
    Génère une structure de densité depuis la surface en suivant un chemin aléatoire dans le plan XY.
    La densité est maximale au centre du chemin (en surface) et décroît latéralement (distance au chemin)
    ainsi qu'en profondeur (distance à la surface).

    Paramètres
    ----------
    density_field : ndarray 3D modifiable
    x, y, z : ndarrays 3D des coordonnées des centres des voxels
    mask : ndarray 3D booléen (True = voxel du volcan)
    surface_func : fonction surface(x, y) -> z_surface (peut être un DEM interpolé)
    center_xy : tuple (cx, cy) point de départ du chemin
    path_radius : rayon (m) du chemin (demi-largeur)
    depth_max : profondeur maximale (m) sous la surface (peut varier le long du chemin)
    density_center : densité au centre du chemin en surface
    background_density : densité de fond
    horizontal_decay : 'linear', 'quadratic', 'gaussian' (décroissance latérale)
    vertical_decay : 'linear', 'quadratic', 'gaussian' (décroissance en profondeur)
    n_points_path : nombre de points pour discrétiser le chemin
    seed : pour reproductibilité
    """
    if seed is not None:
        np.random.seed(seed)

    # Génération d'un chemin 2D aléatoire autour de center_xy
    # On crée une série de points (x_path, y_path) avec une marche aléatoire
    angles = np.random.uniform(0, 2*np.pi, n_points_path)
    steps = np.random.uniform(10, 50, n_points_path)  # longueur des segments (m)
    x_path = [center_xy[0]]
    y_path = [center_xy[1]]
    for i in range(1, n_points_path):
        x_path.append(x_path[-1] + steps[i-1] * np.cos(angles[i-1]))
        y_path.append(y_path[-1] + steps[i-1] * np.sin(angles[i-1]))
    x_path = np.array(x_path)
    y_path = np.array(y_path)

    # Création d'un arbre KD pour le chemin
    path_points = np.vstack((x_path, y_path)).T
    tree = KDTree(path_points)
    # Parcours des voxels
    nx, ny, nz = density_field.shape
    for i in range(nx):
        for j in range(ny):
            xi = x[i, j, 0]
            yj = y[i, j, 0]
            if k_surface[i, j] < 0:
                continue
            # Distance horizontale au chemin
            dist_h, idx = tree.query([xi, yj])
            if dist_h > path_radius:
                continue  # trop loin du chemin
            # Facteur de décroissance horizontale
            if horizontal_decay == 'linear':
                factor_h = 1 - dist_h / path_radius
            elif horizontal_decay == 'quadratic':
                factor_h = 1 - (dist_h / path_radius)**2
            elif horizontal_decay == 'gaussian':
                factor_h = np.exp(- (dist_h / path_radius)**2)
            else:
                factor_h = 1 - dist_h / path_radius
            z_surf = z_surface[i,j]
            # Pour chaque profondeur sous cette colonne
            for k in range(nz):
                if not mask[i, j, k]:
                    continue
                zk = z[i, j, k]
                if zk > z_surf:
                    continue  # au‑dessus de la surface (ne devrait pas arriver)
                depth = z_surf - zk
                if depth > depth_max:
                    continue
                # Facteur de décroissance verticale
                if vertical_decay == 'linear':
                    factor_v = 1 - depth / depth_max
                elif vertical_decay == 'quadratic':
                    factor_v = 1 - (depth / depth_max)**2
                elif vertical_decay == 'gaussian':
                    factor_v = np.exp(- (depth / depth_max)**2)
                else:
                    factor_v = 1 - depth / depth_max

                # Combinaison des facteurs (produit)
                factor = factor_h * factor_v
                new_val = background_density + (density_center - background_density) * factor
                # Mise à jour (on peut prendre le max si plusieurs chemins se superposent)
                if new_val > density_field[i, j, k]:
                    density_field[i, j, k] = new_val
    return

def compute_surface_from_mask(mask_3d, z_3d):
    """
    Calcule pour chaque (i,j) l'altitude du voxel de surface (le plus haut) et son indice k.
    
    Paramètres
    ----------
    mask_3d : ndarray booléen de forme (nx, ny, nz)
    z_3d : ndarray de même forme, coordonnées z des centres des voxels
    
    Retourne
    --------
    z_sruface : ndarray (nx, ny) altitude de surface
    surface_k : ndarray (nx, ny) indice k du voxel de surface
    """
    nx, ny, nz = mask_3d.shape
    surface_k = np.zeros((nx, ny), dtype=int)
    z_sruface = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            # Trouver les k où le masque est vrai
            valid_k = np.where(mask_3d[i, j, :])[0]
            if len(valid_k) > 0:
                k_top = valid_k[-1]  # indice le plus grand (car z croît avec k)
                surface_k[i, j] = k_top
                z_sruface[i, j] = z_3d[i, j, k_top]
            else:
                surface_k[i, j] = -1  # pas de voxel actif dans cette colonne
                z_sruface[i, j] = np.nan
    return z_sruface, surface_k


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Génération aléatoire des modèles
    # ----------------------------------------------------------------------
    survey_name = CURRENT_SURVEY.name   
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"
    vs = int(sys.argv[1]) if len(sys.argv) > 1 else 32  # voxel size in m (edge length)
    # input_vtk = dir_model / f"ElecCond_topo_voi_vox{vs}m.vts"
    input_vtk = dir_voxel / f"topo_voi_vox{vs}m.vts"          # Fichier VTK d'entrée (grille structurée avec masque)
    dir_out = dir_model / "postreg" / "dataset"      # Dossier de sortie
    dir_out.mkdir(parents=True, exist_ok=True)
    log_file = dir_out / f"log_s{SEED}.json"
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Lecture de {input_vtk}...")
    grid, geom = load_voxel_grid(input_vtk)
    mask = geom.mask_voxel
    nx, ny, nz = len(geom.x_edges),len(geom.y_edges),len(geom.z_edges)
    nvx, nvy, nvz = nx-1, ny-1, nz-1


    voxel_coords = get_coordinates(geom)
    xc, yc, zc = voxel_coords.XYZ.T
    # print(check_array_order(xc), check_array_order(mask))
    xc=xc.reshape((nvx,nvy,nvz), order='F')
    # print(xc[:1])
    yc=yc.reshape((nvx,nvy,nvz), order='F')
    zc=zc.reshape((nvx,nvy,nvz), order='F')
    cx,cy,cz = voxel_coords.centre.T
    center_xy=(cx, cy)

    mask = mask.reshape((nvx,nvy,nvz), order='F').astype(bool)
    active = np.where(mask)[0]
    # print(check_array_order(xc), check_array_order(mask))

    file_dem = dir_dem / "topo_roi.vts"
    interp = build_interpolator_topography(file=file_dem)
    xu, yu = xc[:,:,0], yc[:,:,0]
    z_interp = interpolate_topography(interp, xu, yu) #

    z_surface, k_surface = compute_surface_from_mask(mask, zc)
   
    # Plage des coordonnées
    z_min, z_max = np.min(zc[mask]), np.max(zc[mask])
    z_range = (z_min, z_max)  # éviter les bords extrêmes

    log_entries = []
    
    for model_idx in tqdm(range(N_MODELS), total=N_MODELS, desc="Generation"):

        # Initialiser avec la densité de fond
        density = np.full_like(xc, BACKGROUND_DENSITY, dtype=np.float32)
        # density = np.random.uniform(*BACKGROUND_DENSITY_RANGE, size=xc.shape)
        # Appliquer le masque (hors volcan = 0)
        # print(np.count_nonzero(density))
        density[~mask] = 0.0
        # print(np.count_nonzero(density))

        # Nombre de structures (1 à 3)
        n_structures = random.randint(10, 30)

        # Liste des structures de ce modèle (pour log)
        structures_log  = []

        for s in range(n_structures):
            # Choisir un type de structure
            struct_type = random.choice(['non-altered', 'plume'])# 'dike', 'cavity', 'altered',  ]) #
            # print(f"  Structure {s+1} : {struct_type}")

            # Dictionnaire pour cette structure
            struct_info = {'type': struct_type}

            # generate_high_density_from_surface(density, xc, yc, zc, mask,
            #                                 interp, 2.7, BACKGROUND_DENSITY+0.05, 
            #                                 depth_min=30, depth_max=200, 
            #                                 )

            if struct_type == 'non-altered':
                # Centre dans la zone centrale
                center = random_point_in_cylinder(center_xy, RADIUS_CENTRAL, z_range)
                # Tirer un point de surface aléatoire dans le domaine
                x_surf, y_surf = center[0], center[1]
                # x_surf, y_surf = center_xy[0], center_xy[1]
                z_surf = interpolate_topography(interp, x_surf, y_surf)  # à définir selon votre DEM
                surface_point = (x_surf, y_surf, z_surf)

                depth = random.uniform(50, 300)  # profondeur de la région
                radius = random.uniform(50, 200)
                density_center = random.uniform(*NON_ALTERED_DENSITY_RANGE)
                generate_surface_feature_from_path(density, xc, yc, zc, mask,
                                                   z_surface, k_surface, surface_point, 
                                                   radius, depth, 
                                                   density_center, BACKGROUND_DENSITY)
                struct_info.update({
                    'surface_point': [float(surface_point[0]), float(surface_point[1]), float(surface_point[2])],
                    'radius': float(radius),
                    'depth': float(depth),
                    'density': NON_ALTERED_DENSITY_RANGE
                })

            elif struct_type == 'plume':
                # z_bottom = random.uniform(z_range[0]-100, z_range[0])
                z_bottom = z_range[0]
                z_top = random.uniform(z_range[1]-100, z_range[1]-30)
                x_bottom, y_bottom = random_point_in_cylinder(center_xy, RADIUS_CENTRAL, (0,0))[:2]
                bottom = (x_bottom, y_bottom, z_bottom)
                # bottom = (center_xy[0], center_xy[1], z_bottom)
                # dx = random.uniform(-100, 100)
                # dy = random.uniform(-100, 100)
                # top = (xc0 + dx, yc0 + dy, z_top)
                # x_top, y_top = x_bottom+ dx,  y_bottom +  dy
                # top = (x_top, y_top, z_top)
                # radius = random.uniform(*PLUME_RADIUS_RANGE)
                center_density = random.uniform(*PLUME_DENSITY_RANGE)
                # generate_plume_ellipsoid(density, xc, yc, zc, mask,
                #                         bottom, top, radius,
                #                         center_density, BACKGROUND_DENSITY-0.05,
                #                         profile='linear', power=2)
                r0, r1 = random.uniform(50, 150), random.uniform(50, 100)
                depth = random.uniform(30, 100)
                generate_winding_plume(density, xc, yc, zc, mask,
                                       bottom, depth, interp, R0=r0, R1=r1, seed=SEED )
                
                
                struct_info.update({
                    'x_bottom': float(x_bottom),
                    'y_bottom': float(y_bottom),
                    'z_bottom': float(z_bottom),
                    'z_top': float(z_top),
                    # 'radius': float(radius),
                    'density': PLUME_DENSITY_RANGE
                })


            elif struct_type == 'dike':
                # Dyke vertical (dip proche de 0) - centré
                center = random_point_in_cylinder(center_xy, RADIUS_CENTRAL, z_range)
                strike = random.uniform(0, 180)
                dip = random.uniform(-10, 10)  # quasi-vertical
                thickness = random.uniform(*DIKE_THICKNESS_RANGE)
                length = random.uniform(*DIKE_LENGTH_RANGE)
                generate_dike(density, xc, yc, zc, mask, DIKE_DENSITY_RANGE,
                              center, strike, dip, thickness, length)
                struct_info.update({
                    'center': [float(center[0]), float(center[1]), float(center[2])],
                    'strike': float(strike),
                    'dip': float(dip),
                    'thickness': float(thickness),
                    'length': float(length),
                    'density': DIKE_DENSITY_RANGE
                })

            elif struct_type == 'cavity':
                center = random_point_in_cylinder(center_xy, RADIUS_CENTRAL, z_range)
                radius = random.uniform(*CAVITY_RADIUS_RANGE)
                generate_cavity(density, xc, yc, zc, mask, CAVITY_DENSITY_RANGE,
                                center, radius)
                struct_info.update({
                    'center': [float(center[0]), float(center[1]), float(center[2])],
                    'radius': float(radius),
                    'density': CAVITY_DENSITY_RANGE
                })

            elif struct_type == 'altered':
                center = random_point_in_cylinder(center_xy, RADIUS_CENTRAL, z_range)
                rx = random.uniform(*ALTERED_RADIUS_RANGE)
                ry = random.uniform(*ALTERED_RADIUS_RANGE)
                rz = random.uniform(*ALTERED_RADIUS_RANGE) * 0.5  # plus plat verticalement
                generate_altered_zone(density, xc, yc, zc, mask, ALTERED_DENSITY_RANGE,                            
                                         center, (rx, ry, rz))
                struct_info.update({
                    'center': [float(center[0]), float(center[1]), float(center[2])],
                    'radii': [float(rx), float(ry), float(rz)],
                    'density': ALTERED_DENSITY_RANGE
                })

            
            structures_log.append(struct_info)
        # density[mask] = smooth_model_gaussian(density[mask], sigma=0.5)
        # Sauvegarde VTK
        vtk_fname = dir_out / f"model_{model_idx:03d}.vts"
        save_model_vtk(grid, density, vtk_fname, "density")
        # print(f"  VTK sauvegardé dans {vtk_fname}")

        # Sauvegarde NPZ (pour entraînement)
        npz_fname = dir_out / f"model_{model_idx:03d}.npz"
        save_model_npz(density, mask, npz_fname)
        # print(f"  NPZ sauvegardé dans {npz_fname}")

        # Ajouter l'entrée de log pour ce modèle
        log_entries.append({
            'model_index': model_idx,
            'seed': SEED,  # on peut noter le seed global, ou sauvegarder l'état du générateur pour reproductibilité fine
            'structures': structures_log
        })
    
    # Sauvegarde du fichier de log JSON
    with open(log_file, 'w') as f:
        json.dump(log_entries, f, indent=2)
    print(f"Log sauvegardé dans {log_file}")

    print("Terminé.")
