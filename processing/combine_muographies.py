#!/usr/bin/python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
# package module(s)
from config import STRUCT_DIR, DATA_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime, cm_batlow
from func import test_run_key, set_norm

titlesize=36
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


survey_name = CURRENT_SURVEY.name   
struct_dir = STRUCT_DIR / survey_name

dirs = {
        "structure":{
                "voxel": struct_dir/"voxel",
                "tel": struct_dir/"telescope",
                "flux": struct_dir/"flux",
            },
       "data":  DATA_DIR,
        } 

dtel = CURRENT_SURVEY.telescopes

nim = 3
nconf = 0
for _, tel in dtel.items():  nconf += len(tel.configurations)
nrows, ncols = nconf//2, nim * 2
fig0, axs0 = plt.subplots(figsize=(8*ncols, 7*nrows), 
                        nrows=nrows, 
                        ncols=ncols, 
                        # constrained_layout=True, 
                        sharex=True, 
                        sharey=True)

fig0.text(x=0.5, y=0.92 - 0.1 / (nrows + 1), 
                    s="",
                    fontsize=1.5*titlesize,
                    rotation='horizontal',
                    color = 'black',
                    # fontweight='bold',
                    ha='center')

fig1, axs1 = plt.subplots(figsize=(8*ncols, 7*nrows), 
                        nrows=nrows, 
                        ncols=ncols, 
                        # constrained_layout=True, 
                        sharex=True, 
                        sharey=True)

fig1.text(x=0.5, y=0.92 - 0.1 / (nrows + 1), 
                    s="",
                    fontsize=1.5*titlesize,
                    rotation='horizontal',
                    color = 'black',
                    # fontweight='bold',
                    ha='center')

r, c = 0, 0


for tel_name, tel in tqdm(dtel.items(), desc="Muography"):
    
    h5file_muo = dirs["data"] / tel.name / f"muography.h5"
    
    with h5py.File(h5file_muo) as fh5:
        pass

        for conf_name, conf in tel.configurations.items():
            pass
            if r == nconf//2 :  r, c = 0, nim

            u_edges, v_edges = conf.u_edges, conf.v_edges
            u, v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
            nu, nv = conf.shape_uv

            run_name = test_run_key(tel_name, fh5)  
            # print(fh5[tel_name][run_name][conf_name]["rays_length"][0].shape)
            rays_length, unc_rays_length = np.array(fh5[tel_name][run_name][conf_name]["rays_length"])[0]
            rays_length = np.array(rays_length).reshape(nu, nv)
            # print(f"{tel_name} {conf_name} rays_length min, max: ", np.nanmin(rays_length), np.nanmax(rays_length))
            counts, unc_counts = np.array(fh5[tel_name][run_name][conf_name]["counts"])[0]
            flux, unc_flux = np.array(fh5[tel_name][run_name][conf_name]["flux"])[0]
            opacity, unc_opacity = np.array(fh5[tel_name][run_name][conf_name]["opacity"])[0]
            mask = (counts > 0) & np.isfinite(counts) & (1e1 < rays_length) & (rays_length<=1.e3) 
            counts = np.array(counts)
            counts[~mask] = np.nan
            flux = np.array(flux)
            flux[~mask] = np.nan
            opacity = np.array(opacity)
            opacity[~mask.ravel()] = np.nan 
            
            _ax = axs0[r, c] 
            if r==0 : _ax.set_title(f"Counts") 
            array = counts.reshape(nu, nv)
            norm_counts = LogNorm(vmin=1, vmax=1e3) if len(tel.panels) < 4 else LogNorm(vmin=1, vmax=1e4)
            array = np.flipud(array) if tel.flipped else array
            im0 = _ax.pcolormesh(u,v, array, shading='auto',  cmap='viridis',norm=set_norm(array))
            cax0 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb0= fig0.colorbar(im0, cax=cax0, extend='max')
            _ax.text(0.05, 0.92, f"{tel.name}_{conf_name}", fontsize="xx-large", color="black", 
                            transform=_ax.transAxes, 
                            bbox= props,
                            ha="left",va="bottom",**{"fontweight":"bold"})            
            _ax = axs0[r,c+1]
            if r==0 : _ax.set_title(f"Flux") 
            array = flux.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            norm_flux = LogNorm(vmin=1e-6, vmax=1e-3)
            im1 = _ax.pcolormesh(u, v, array, shading='auto', norm=norm_flux, cmap="RdBu")
            cax1 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb1= fig0.colorbar(im1, cax=cax1, extend='max')
            
            _ax = axs0[r,c+2]
            if r==0 : _ax.set_title(f"Opacity") 
            array = opacity.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            norm_opacity = LogNorm(vmin=1e1, vmax=3e3)
            im2 = _ax.pcolormesh(u, v, array, shading='auto', norm=norm_opacity, cmap=cm_batlow)
            cax2 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb2= fig0.colorbar(im2, cax=cax2, extend='max')  
            for _a in axs0.ravel(): 
                _a.set_xlabel("tan($\\theta_x$)")
                _a.set_ylabel("tan($\\theta_y$)")
                _a.label_outer()
            
            _ax = axs1[r, c] 
            if r==0 : _ax.set_title(f"Statistical Uncertainty Counts") 
            array = unc_counts.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            im0 = _ax.pcolormesh(u,v, array, shading='auto',  cmap='viridis',norm=set_norm(array))
            cax0 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb0= fig1.colorbar(im0, cax=cax0, extend='max')
            _ax.text(0.05, 0.92, f"{tel.name}_{conf_name}", fontsize="xx-large", color="black", 
                            transform=_ax.transAxes, 
                            bbox= props,
                            ha="left",va="bottom",**{"fontweight":"bold"})            
            _ax = axs1[r,c+1]
            if r==0 : _ax.set_title(f"Rel. Unc. Flux") 
            rel_unc_flux = np.zeros_like(unc_flux)
            mask_flux = (flux > 0) & np.isfinite(flux) & (unc_flux > 0) & np.isfinite(unc_flux)
            rel_unc_flux[mask_flux] = unc_flux[mask_flux] / flux[mask_flux]
            array = rel_unc_flux.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            im1 = _ax.pcolormesh(u, v, array, shading='auto', norm=LogNorm(vmin=1e-2, vmax=1e0), cmap="RdBu")
            cax1 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb1= fig1.colorbar(im1, cax=cax1, extend='max')
            
            _ax = axs1[r,c+2]
            if r==0 : _ax.set_title(f"Rel. Unc. Opacity") 
            rel_unc_opacity = np.zeros_like(unc_opacity)
            mask_opacity = (opacity > 0) & np.isfinite(opacity) & (unc_opacity > 0) & np.isfinite(unc_opacity)
            rel_unc_opacity[mask_opacity] = unc_opacity[mask_opacity] / opacity[mask_opacity]
            array = rel_unc_opacity.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            im2 = _ax.pcolormesh(u, v, array, shading='auto', norm=LogNorm(vmin=1e-2, vmax=1e0), cmap=cm_batlow)
            cax2 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb2= fig1.colorbar(im2, cax=cax2, extend='max')  
            for _a in axs1.ravel(): 
                _a.set_xlabel("tan($\\theta_x$)")
                _a.set_ylabel("tan($\\theta_y$)")
                _a.label_outer()


            
            r+=1
            
strdate = time.strftime("%d%m%Y")
dout = Path(__file__).parent / "png"
dout.mkdir(exist_ok=True, parents=True) 
fout0 = dout / f"muography_combined_{strdate}.png"
fig0.savefig(fout0)
print(f"Saved {fout0}")

fout1 = dout / f"uncertainty_muography_combined_{strdate}.png"
fig1.savefig(fout1)
print(f"Saved {fout1}")

exit()

nim = 2
nconf = 0
for _, tel in dtel.items():  nconf += len(tel.configurations)
nrows, ncols = nconf//2, nim * 2
fig, axs = plt.subplots(figsize=(8*ncols, 7*nrows), 
                        nrows=nrows, 
                        ncols=ncols, 
                        # constrained_layout=True, 
                        sharex=True, 
                        sharey=True)
fig.text(x=0.5, y=0.92 - 0.1 / (nrows + 1), 
                    s="",
                    fontsize=1.5*titlesize,
                    rotation='horizontal',
                    color = 'black',
                    # fontweight='bold',
                    ha='center')
r, c = 0, 0

run_name = "calib"

for tel_name, tel in tqdm(dtel.items(), desc="Acceptance"):
    
    h5file_muo = dirs["data"] / tel.name / f"muography.h5"
    
    with h5py.File(h5file_muo) as fh5:
        pass

        for conf_name, conf in tel.configurations.items():
            pass
            if r == nconf//2 :  r, c = 0, nim
            
        
            u_edges, v_edges = conf.u_edges, conf.v_edges
            u, v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
            nu, nv = conf.shape_uv


            counts = fh5[tel_name][run_name][conf_name]["counts"]
            counts = np.array(counts)
            unc_counts = np.sqrt(counts)
            acc, unc_acc = np.array(fh5[tel_name][run_name][conf_name]["acceptance"])[0]
            # opacity = fh5[tel_name][run_name][conf_name]["opacity"]
            # opacity = np.array(opacity)

            _ax = axs[r, c] 
            if r==0 : _ax.set_title(f"Counts") 
            array = counts.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            im0 = _ax.pcolormesh(u,v, array, shading='auto', norm=set_norm(array), cmap='viridis')
            cax0 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb0= fig.colorbar(im0, cax=cax0, extend='max')
            _ax.text(0.05, 0.92, f"{tel.name}_{conf_name}", fontsize="xx-large", color="black", 
                            transform=_ax.transAxes, 
                            bbox= props,
                            ha="left",va="bottom",**{"fontweight":"bold"})            

            _ax = axs[r,c+1]
            if r==0 : _ax.set_title(f"Acceptance") 
            array = acc.reshape(nu, nv)
            array = np.flipud(array) if tel.flipped else array
            im1 = _ax.pcolormesh(u, v, array, shading='auto', norm=set_norm(array), cmap="RdBu")
            cax1 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            cb1= fig.colorbar(im1, cax=cax1, extend='max')
            
            # _ax = axs[r,c+2]
            # if r==0 : _ax.set_title(f"Opacity") 
            # array = opacity.reshape(nu, nv)
            # array = np.flipud(array) if tel.flipped else array
            # im2 = _ax.pcolormesh(u, v, array, shading='auto', norm=set_norm(array), cmap=cm_batlow)
            # cax2 = inset_locator.inset_axes(_ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
            # cb2= fig.colorbar(im2, cax=cax2, extend='max')  
            # for _a in axs.ravel(): 
            #     _a.set_xlabel("tan($\\theta_x$)")
            #     _a.set_ylabel("tan($\\theta_y$)")
            #     _a.label_outer()
            r+=1

dout = Path(__file__).parent / "png"
dout.mkdir(exist_ok=True, parents=True) 
fout = dout / f"acceptance_combined_{strdate}.png"
fig.savefig(fout)
print(f"Saved {fout}")