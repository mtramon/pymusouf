#!/usr/bin/python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np
import pickle
import scipy.sparse as sp
import sys
from tqdm import tqdm
import vtk
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
from voxelgrid import load_voxel_grid

titlesize=24
params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':titlesize,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':1,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': False,
         'figure.figsize': (8,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
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
    vs = int(sys.argv[1]) if len(sys.argv) >1 else 32  # voxel size in m (edge length)
    # input_file = dir_voxel / f"topo_voi_vox{vs}m.vts"
    input_file = dir_voxel / f"topo_bulge_voi_vox{vs}m.vts"
    # input_file = dir_model / f"ElecCond_CentralCube_aligned_voi_vox{vs}m.vts"
    input_file = dir_model / f"ElecCond_topo_voi_vox{vs}m.vts"
    # input_file = dir_voxel / f"topo_center_anom_voi_vox{vs}m.vts"

    print_file_datetime(input_file)
    
    ## Read source structured grid
    grid, geom = load_voxel_grid(input_file)
    voxel_density = geom.density
    # rho = voxel_density.reshape(-1, order="F")
    rho=voxel_density.copy()
    # print(f"Read voxel volumes and densities from VTK file")
    x_edges, y_edges, z_edges = geom.x_edges, geom.y_edges, geom.z_edges
    nx, ny, nz = len(x_edges), len(y_edges), len(z_edges)
    nvx, nvy, nvz = nx-1, ny-1, nz-1
    nvox = nvx*nvy*nvz
    mask_voi = np.ones(nvox)
    mask_voi = (rho > 0.0)   # True = voxel du volcan
    mask_voi = mask_voi.astype(np.uint8)  # Numba-friendly

    x =(x_edges[:-1] + x_edges[1:])/2
    y =(y_edges[:-1] + y_edges[1:])/2
    z =(z_edges[:-1] + z_edges[1:])/2
    cx =(x_edges[0] + x_edges[-1])/2
    cy =(y_edges[0] + y_edges[-1])/2
    cz =(z_edges[0] + z_edges[-1])/2

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)    

    ORDER = "F" #each voxel volume mask needs to be in Fortran order (to match M matrix building)
    mask_center = (R <= 300)
    mask_center = mask_center.reshape(-1, order=ORDER)
    zmax = np.max(z)
    mask_z = (Z > abs(zmax-300))
    mask_z = mask_z.reshape(-1, order=ORDER)
    # mask_voxel = mask_voi & mask_center & mask_z
        
    rho_0 = 2700 # in kg/m^3
    D_0 = rho_0 * np.ones(nvox, dtype=np.float64)  # density vector in kg/m^3
    D_0 = D_0 * mask_voi
    D_1 = voxel_density * mask_voi  
    if D_1.max() < 1e1: D_1 *= 1000 # if density is in g/cm^3, convert to kg/m^3
    
    basename = f"real_telescopes"   
    dtel = CURRENT_SURVEY.telescopes

    # basename = f"toy_telescopes_s9506"   
    # fin_toytel = dir_tel / f"toy_telescopes_s9506_vox{vs}m.pkl"
    # with open(fin_toytel, 'rb') as f:
    #     dtel = pickle.load(f) 

    h5_path = dir_voxel / f"{basename}_voxel_ray_matrices_vox{vs}m.h5"

    nconf = 0
    for _, tel in dtel.items():  nconf += len(tel.configurations)
    nim = 4 
    nrows, ncols = nconf//2, nim * 2
    fig, axs = plt.subplots(figsize=(8*ncols, 7*nrows), 
                            nrows=nrows, 
                            ncols=ncols, 
                            # constrained_layout=True, 
                            sharex=True, 
                            sharey=True)
    
    fig.text(x=0.5, y=0.92 - 0.1 / (nrows + 1), 
                        s=f"Opacity \n Homogeneous model $\\rho_0$ = {2.7} g/cm$^3$ ; Conductivity model $\\rho_m$ $\\in$ [{np.min(D_1[D_1!=0])*1e-3:.1f}; {np.max(D_1)*1e-3:.1f}] g/cm$^3$ ; Relative variation $\\Delta \\varrho$ ",
                        fontsize=1.5*titlesize,
                        rotation='horizontal',
                        color = 'black',
                        # fontweight='bold',
                        ha='center')
    r, c = 0, 0
    # norm_opacity = LogNorm(1, 3e3) #mwe
    norm_opacity = LogNorm(1e3, 3e6) #kg/m^2
    norm_length = LogNorm(1, 1e3)  #m
    norm_variation = Normalize(0.02, 0.2)
    mask_weight = np.ones_like(mask_voi)
    str_unit =  "kg/m$^2$" #"mwe" 
    with h5py.File(h5_path) as fh5:
        for tel_name, tel in tqdm(dtel.items(), desc="Opacity"):
            tel.compute_angular_coordinates()
            # if tel.name != "SNJ": continue
            # x0, y0, z0 = tel.coordinates
            for conf_name, conf in tel.configurations.items():
                if r == nconf//2 :  r, c = 0, nim
                ze_matrix = tel.zenith_matrix[conf_name]
                mask_rays = (ze_matrix <= 90 * np.pi/180)
                nu, nv = conf.shape_uv
                # if conf_name != "4p" and conf_name != "3p1": continue
                u_edges, v_edges = conf.u_edges, conf.v_edges
                u, v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
                # size_panel = conf.panels[0].matrix.scintillator.length * 1e-3  # mm-> m
                # G_uv = geometrical_acceptance(u_edges=u_edges, v_edges=v_edges, delta_z = conf.length_z*1e-3, Lx =size_panel, Ly =size_panel, nu_sr=10, nv_sr=10)
                dtel = CURRENT_SURVEY.telescopes
                # voxray_file = dir_voxel / f"voxel_ray_matrix_{tel.name}_{conf_name}_vox{vs}m.npz"
                # voxray = np.load(voxray_file, allow_pickle=True) 
                voxray = fh5[tel_name][conf_name]
                # Extraire les attributs
                weights = voxray["data"]
                indices = voxray["indices"]
                indptr = voxray["indptr"]
                shape = voxray["shape"]
                nvox, npix = shape
                G_uv = np.array(voxray["acceptance"])
                lengths = np.array(voxray["rays_length"])
                # Reconstruire la matrice CSR
                M = sp.csr_matrix(
                    (weights, indices,indptr),
                    shape=shape
                ).T
                print(type(M))
                # mask_weight = (weight_sum >= 0).ravel(ORDER)    
                mask_voxel = mask_voi  & mask_z & mask_center #& mask_weight 

                mask_rays = np.ones(npix, dtype=bool)#

                active = np.where(mask_voxel)[0]
                opaque = np.where(mask_rays)[0]
                M = M.multiply(1.0 / (G_uv[:, None])).tocsc()
                weight_sum = np.array(M.sum(axis=0))
                print(f"{tel_name}, {conf_name}: sum_weight min, max = {np.min(weight_sum):.3e}, {np.max(weight_sum):.3e}")
                length_sum = np.array(M.sum(axis=1))
                print(f"{tel_name}, {conf_name}: ray_length min, max = {np.min(length_sum):.3e}, {np.max(length_sum):.3e}")
                # M = M.multiply(1.0 /(G_uv[:, None]) ) 
                M = M[opaque][:, active] 
                
                O_0 = np.zeros(npix)
                # O_0[opaque] = M.dot(D_0[active])  #.reshape(nu, nv) #* mask_rays   # opacity vector shape (npix,)
                O_0[opaque] = M.dot(D_0[active]) #[mask_rays]
                # O_0[opaque] = M.dot(D_0[active]) 
                # O_0 = M.dot(D_0) #* lengths 
                # print(np.nanmean(O_0[mask_rays]))
                # O_0 /= G_uv
                # O_0 *= 1e-3 ##kg/m^2 -> mwe
                # O_0 = O_0
                
                O_1 = np.zeros(npix)
                # O_1[opaque] = M.dot(D_1[active])   #* mask_rays 
                # O_1[mask_rays] = M.dot(D_1) [mask_rays]
                O_1[opaque] = M.dot(D_1[active])
                # O_1 = M.dot(D_1)
                # O_1 /= G_uv
                # O_1 *= 1e-3 ##kg/m^2 -> mwe
                # unc = np.random.normal(0.2*O_1, 0.1*O_1)

                mask_rays = mask_rays & (O_1 > 1e4)

                
                unc = np.zeros_like(O_1)
                mean_opacity, std_opacity = np.mean(O_1[mask_rays]), np.std(O_1[mask_rays])

                # op_scaled = O_1[mask_rays]
                # op_scaled = (op_scaled-np.min(op_scaled)) / (np.max(op_scaled)-np.min(op_scaled))
                # unc_model = 1/(1+np.random.normal(0.1*op_rand, 0.05*O_1[mask_rays]))
                # unc_model = 0.025 + 0.07/(1+np.exp(40*(op_scaled-0.02))) + 0.05/(1+np.exp(8*(0.9-op_scaled))) #noise model Lelièvre 2019
                # unc[mask_rays] = np.clip(unc_model, 1e-4*mean_opacity, 0.2*mean_opacity)
                unc[mask_rays] = np.clip(0.01*O_1[mask_rays], 1e-4*mean_opacity, 0.2*mean_opacity)
                # O_1 += unc 
                # O_1 = O_1.reshape(nu, nv)

                vmax_o = np.nanmax(O_0)  # cap max opacity at 1000 kg/m^2 for better color scaling
                _ax = axs[r, c] 
                O_0[~mask_rays] = np.nan
                im0 = _ax.pcolormesh(u,v,O_0.reshape(nu, nv), shading='auto', norm=norm_opacity, cmap='viridis')
        
                _ax.set_title(f"homogeneous, mean = {np.nanmean(O_0):.2e} {str_unit}")

                _ax.text(0.05, 0.92, f"{tel.name}_{conf_name}", fontsize="xx-large", color="black", 
                            transform=_ax.transAxes, 
                            bbox= props,
                            ha="left",va="bottom",**{"fontweight":"bold"})
                cax = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
                cb= fig.colorbar(im0, cax=cax, extend='max')
                
                _ax = axs[r,c+1]
                lengths[~mask_rays] = np.nan
                im1 = _ax.pcolormesh(u, v, lengths.reshape(nu, nv), shading='auto', norm=norm_length, cmap='viridis')
                # im1 = _ax.pcolormesh(u, v, unc.reshape((nu, nv), order=ORDER), shading='auto', norm=norm_opacity, cmap='viridis')
                # _ax.set_title(f"unc, mean = {unc.mean():.2e} {str_unit}")
                cax = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
                cb= fig.colorbar(im1, cax=cax, extend='max')
                _ax.set_title(f"length, mean = {np.nanmean(lengths):.2e} m")

                _ax = axs[r,c+2]
                O_1[~mask_rays] = np.nan
                im1 = _ax.pcolormesh(u, v, O_1.reshape(nu, nv), shading='auto', norm=norm_opacity, cmap='viridis')
                _ax.set_title(f"model+unc, mean = {np.nanmean(O_1):.2e} {str_unit}")
                cax = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
                cb= fig.colorbar(im1, cax=cax, extend='max')
                cb.set_label(f" $\\varrho$ [kg/m$^2$]" , labelpad=1)

                # im2 = axs[2].pcolormesh(u,v, dO, shading='auto', norm=LogNorm(np.nanmin(dO[dO!=0]), np.nanmax(dO)), cmap='RdBu_r')
                _ax = axs[r,c+3]
                cw_map=plt.get_cmap('coolwarm').reversed()
                dO =  abs(O_1 - O_0) / O_0  # relative change in opacity

                # dO[~mask_rays] = np.nan
                im2 =_ax.pcolormesh(u, v, dO.reshape(nu, nv), shading='auto', norm=norm_variation, cmap=cw_map)
                # im2 = _ax.pcolormesh(u,v, dO, shading='auto', norm=LogNorm(0.001, np.nanmax(dO)), cmap=cw_map)
                _ax.set_title(f"mean = {np.nanmean(dO)*100:.1f} %")
                cax = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
                cb= fig.colorbar(im2, cax=cax, extend='max')
                cb.set_label(f" |$\\Delta\\varrho$| / $\\varrho_0$", labelpad=1)
                for _a in axs.ravel(): 
                    _a.set_xlabel("tan($\\theta_x$)")
                    _a.set_ylabel("tan($\\theta_y$)")
                    _a.label_outer()
                r+=1
    cb.ax.tick_params(which="both", labelsize="x-large",pad=1)  
    plt.show()
    # dout = dir_voxel / "png"
    # dout.mkdir(exist_ok=True, parents=True) 
    # fout = dout / f"opacity_all_vox{vs}m.png"
    # fig.savefig(fout)
    # print(f"Saved opacity map to {fout}")