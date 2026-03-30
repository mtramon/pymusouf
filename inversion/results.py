
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np
from pathlib import Path
import sys
# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime
titlesize=24
params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'xx-large',
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


def compute_optimal_lamba(r,s):
    '''
    For L-curve
    '''
    xi = np.log(r)
    eta = np.log(s)
    # Calcul des dérivées premières et secondes (centrées)
    dxi = np.gradient(xi, np.log(lambda_reg))
    deta = np.gradient(eta, np.log(lambda_reg))
    d2xi = np.gradient(dxi, np.log(lambda_reg))
    d2eta = np.gradient(deta, np.log(lambda_reg))
    # Courbure
    courbure = np.abs(dxi * d2eta - deta * d2xi) / (dxi**2 + deta**2)**1.5
    # Indice du maximum (sauf les bords)
    idx_opt = np.argmax(courbure[1:-1]) + 1
    lambda_opt = lambda_reg[idx_opt]
    return lambda_opt,idx_opt


if __name__ == "__main__":



    survey_name = CURRENT_SURVEY.name   
    print(f"Processing survey: {survey_name}")
    print(f"Survey structure directory: {STRUCT_DIR / survey_name}")
    dir_survey = STRUCT_DIR / survey_name
    dir_dem = dir_survey / "dem"
    dir_voxel = dir_survey / "voxel"
    dir_model = dir_survey / "model"
    vs = int(sys.argv[1]) if len(sys.argv) > 1 else 32  # voxel size in m (edge length)

    dir_out = dir_model / "test"
    dir_out.mkdir(parents=True, exist_ok=True)
    # fin = dir_out / f"density_models_vox{vs}m.npz"
    fin = Path(__file__).parent / f"results.npz"
    print_file_datetime(fin)

    arrays = np.load(fin, allow_pickle=True)
    density_post = arrays["density_post"]
    density_true = arrays["density_true"] 
    mask_voxel = arrays["mask_voxel"].astype(bool)
    lambda_reg= arrays["lambda_reg"]
    mu_damp = arrays["mu_damp"] 
    ndata = arrays["ndata"]
    nl = len(lambda_reg)
    nmu = len(mu_damp)
    ntot = nl * nmu
    misfits_data, misfits_model = arrays["misfits"]

    '''
    fig, ax = plt.subplots()
    misfits_tot= misfits_data+misfits_model
    ax.plot(lambda_reg, misfits_data, label="data")
    ax.plot(lambda_reg, misfits_model, label="model")
    ax.plot(lambda_reg, misfits_tot, label="total")
    ax.set_xlim(lambda_reg.min(), lambda_reg.max())
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda_r$")
    ax.set_yscale("log")
    ax.set_ylabel("misfit")
    ax.grid( which="both")
    ax.legend(loc="best")
    fout_png = dir_out / f"misfits_vs_lambda.png"
    fig.savefig(fout_png)
    print(f"Saved {fout_png}")
    '''

    if "sensitivity" in arrays.keys() : 
        sensitivity = arrays["sensitivity"]
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
        fout_png = dir_out / f"sensitivity.png"
        fig.savefig(fout_png)
        print(f"Saved {fout_png}")

    mean_true = np.mean(density_true[mask_voxel])
    min_post, max_post = np.min(density_true[mask_voxel]), np.max(density_true[mask_voxel])
    mean_post = np.mean(density_post[:,mask_voxel], axis=1)
    min_post, max_post = np.min(density_post[:,mask_voxel], axis=1), np.max(density_post[:,mask_voxel], axis=1)

    n = misfits_data.shape[0]
    ncols, nrows = 7, 6
    fig, axs = plt.subplots(figsize=(7*ncols, 6*nrows),ncols=ncols, nrows=nrows)
    # ymin, ymax = min_post, max_post
    # ax.fill_between(lambda_reg, ymin, ymax, alpha=0.5)
    # ax.plot(lambda_reg, mean_post)
    bins=100
    xtrue = density_true[mask_voxel]
    min0, max0 = 0.9e3, 4.1e3#min(xtrue)-500, max(xtrue)+500
    h, e = np.histogram(xtrue, bins=bins, range=[min0, max0],  density=True)
    ymax = np.max(h)
    bc, w = (e[:-1]+e[1:])/2, abs(e[:-1]-e[1:])
    ax0 = axs.ravel()[0]
    ax0.bar(bc,  h, w, color="orange", label = f"True\nmean={mean_true:.3e}")
    ax0.legend(loc="best", fontsize=24)
    ax0.set_xlim(min0, max0)
    # ax0.set_yscale("log")
    ix_opt = np.argmin(abs(misfits_data / ndata - 1))
    for j in range(nmu):
        mu = mu_damp[j]
        for i in range(nl):
            lr = lambda_reg[i] 
            ix = i + j * nl
            ax = axs.ravel()[ix+1]
            xpost = density_post[ix,mask_voxel]
            # _min, _max = min(xpost), max(xpost)
            h, e = np.histogram(xpost, bins=bins, range=[min0, max0], density=True)
            bc, w = (e[:-1]+e[1:])/2, abs(e[:-1]-e[1:])
            if misfits_data[ix] == misfits_data[ix_opt] : color="red"
            else:     color="blue"
            ax.bar(bc,  h, w, color=color, label = f"ix:{ix}\n$\\lambda_r$={lr:.2e}, $\\mu_d$={mu:.1e} \n $\\chi_d$/$N_d$ = {misfits_data[ix] / ndata:.2e}")
            
            ax.set_xlim(min0, max0)
            # ax.set_yscale("log")
            ax.set_ylim(0, ymax)
            # ax.label_outer()
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            # ax.set_xlabel("$\\lambda_r$")
            # ax.set_xlabel("$\\rho$ post [kg/m$^3$]")
            ax.legend(loc="best", fontsize=24)
            ax.label_outer()
    fout_png = dir_out / f"density_post.png"
    fig.savefig(fout_png)
    print(f"Saved {fout_png}")

    fig, ax = plt.subplots()
    ymin, ymax = min_post, max_post
    # for i in range(lambda_reg):
    vmin, vmax=min(lambda_reg), max(lambda_reg)
    norm = LogNorm(vmin,vmax)
    cmap = plt.cm.viridis
    range_val = np.linspace(vmin, vmax, 100)
    color_scale =  cmap(norm(range_val))   
    arg_col =  [np.argmin(abs(range_val-v))for v in lambda_reg]   
    lambda_col = color_scale[arg_col]
    for j in range(nmu):
        mu = float(mu_damp[j])
        ix = np.arange(nl) + j * nl
        chi2_m, chi2_d = misfits_model[ix], misfits_data[ix] / ndata
        r = np.sqrt(chi2_m)
        s = np.sqrt(chi2_d)
        lambda_opt, ix_opt = compute_optimal_lamba(r, s)
        # print(lambda_opt, ix_opt)
        ax.scatter(chi2_d, chi2_m,color=lambda_col)
        ax.plot(chi2_d, chi2_m)
        ax.scatter(chi2_d[ix_opt], chi2_m[ix_opt], marker="*", s=200, label=f"{ix[ix_opt]}:" + " $\\lambda_r ^{opt}$ ($\\mu$="+f"{mu}) =" + f"{lambda_opt:.3e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("$\\chi_d$ / $N_d$")
    ax.set_ylabel("$\\chi_m$")
    cax = inset_locator.inset_axes(ax,
            width="5%",  
            height="100%",
            loc='right',
            borderpad=-4
        )
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,  extend='max') # pad=0.01, shrink=0.9,
    cbar.set_label('$\\lambda_r$', labelpad=5)  
    ax.legend(loc="best", fontsize=24)
    fout_png = dir_out / f"misfits_model_vs_data.png"
    fig.savefig(fout_png)
    print(f"Saved {fout_png}")


    