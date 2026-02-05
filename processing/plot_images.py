import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import numpy as np 
import pandas as pd
import pickle

#package modules
from acceptance.acceptance import GeometricalAcceptance
from cli import get_common_args
from data import RawData
from reco.eventrate import EventRate
from utils.common import Common
from utils.tools import print_file_datetime

params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':1,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': False,
         'figure.figsize': (8,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
plt.rcParams.update(params)


def correct_tomo_images(htomo, hcalib, binx=None, biny=None, eps=1e-12):
    m = hcalib != 0 
    htomo[~m] = np.nan
    htomo[m] /= hcalib[m]
#    htomo /= (hcalib+eps)
    return htomo

if __name__ == "__main__":

    args = get_common_args(save=False)
    cmn = Common(args)
    survey = cmn.survey
    tel = cmn.telescope
    runs = survey.runs[tel.name]
    dict_conf = tel.configurations
    
    tel.compute_angle_matrix()
    file_track = cmn.reco_path / "df_track.csv.gz"
    print_file_datetime(file_track)
    dir_out = cmn.plot_path
    df = pd.read_csv(file_track)
    df.set_index("event_id", inplace=True)
    
    # fig, ax = plt.subplots(layout="compressed")
    # ax.hist(df["rms"], bins=100)
    # fout=dir_out/"rms.png"
    # fig.savefig(fout)
    # print(f"Save {fout}")
    # print(len(df.groupby('config')))
    grp_evt_id = df.groupby('event_id')
    # print(len(grp_evt_id))
    t = grp_evt_id['timestamp'].first().to_numpy()
    er = EventRate(time=t, t0=0)
    fig, ax = plt.subplots(figsize=(12,9))  
    window = 12 if "tomo" in args.run else 3
    _, rate = er.time_series(ax, width=3600, window=window, label="", **{"alpha":1., "linewidth":1, "color":tel.color})
    ax.set_xlim(0, ax.get_xticks()[-1])
    ax.set_ylim(0, ax.get_yticks()[-1])
    nev = len(t)
    ax.text( 0.82,0.95,
            f"nev={nev:.2e}", 
            fontsize="xx-large", ha="left",va="bottom", transform=ax.transAxes, # fontweight="bold",
        )
    fout = cmn.plot_path / f"eventrate.png"
    ax.grid(alpha=0.2)
    fig.savefig(fout)
    print(f"Save {fout}")
    # exit()
    ncols,nrows = len(dict_conf),2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), constrained_layout=True,  sharex=True, sharey=True) #gridspec_kw=kwargs_size)
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    if axs.ndim ==1 : axs=axs[:,np.newaxis]
    vmax = 0
    dict_out = {}
    l_h2d = []
    for i in range(2):
        fin = i 
        for j,(c, conf) in enumerate(dict_conf.items()):
            mask = (df['config'] == c) 
            if fin == 1 : mask = mask & (df['inside'] == fin)
            x, y = df["dx_dz"][mask], df["dy_dz"][mask]
            # range_xy = np.array([[min(x), max(x)], [min(y), max(y)]])
            # if fin == 1: print(range_xy, conf.range_tanthetaxy)
            range_xy = conf.range_tanthetaxy
            # print(range_xy)
            shp = tel.azimuth_matrix[c].shape
            bins = shp #
            # if i==1 and c.startswith("3p"): bins = (shp[0]-10, shp[1]-10)
            # dz = conf.length_z 
            # range_xy = np.arange(-15.5, 16.5)
            # bins = (range_xy * 50) / dz
            # if c == "4p": 

            #     x = x + np.random.uniform(-1e-2, 1e-2, size=len(x))
            #     y = y + np.random.uniform(-1e-2, 1e-2, size=len(y))
            h, binx, biny = np.histogram2d(x, y, bins=bins, range=range_xy)
            if i == 0 : dict_out[c] = {"h":h,"binx":binx,"biny":biny}
            _max = np.max(h) 
            vmax= _max if vmax < _max else vmax
            if i == 0 :axs[i,j].set_title(c + " config")
            l_h2d.append((h, binx, biny))
    fout = cmn.pkl_path / "images_brut.pkl"
    with open(fout, 'wb') as f:
        pickle.dump(dict_out, f)
    print(f"Save {fout}")
    

    for i, ax in enumerate(axs.ravel()):
        h, bx, by = l_h2d[i]
        h = np.flipud(h) if (tel.name == "SXF") or (tel.name == "OM") else h
        # im = ax.imshow(h, norm=LogNorm(1, vmax))
        im = ax.pcolormesh(bx, by, h, norm=LogNorm(1, vmax))
        if i == len(axs.ravel())-1:
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.1)
            cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-3,loc = 'right')
            cb= fig.colorbar(im, cax=cax, extend='max')
            cb.set_label(f'Entries', labelpad=1)
            cb.ax.tick_params(which="both", labelsize="x-large",pad=1)    
        # ax.set_aspect("equal")
        ax.set_xlabel("tan($\\theta_x$)")
        ax.set_ylabel("tan($\\theta_y$)")
        ax.label_outer()
    # fig.tight_layout()
    file_out = dir_out / "images_brut.png"
    fig.savefig(file_out, bbox_inches="tight", dpi=200)
    print(f"Save {file_out}")

    if not "calib" in runs.keys(): exit()
    run_calib = runs.get("calib")
    pkl_calib = run_calib.path / "pkl" / "images_brut.pkl"
    if pkl_calib.exists() and "tomo" in args.run: 
        with open(pkl_calib, 'rb') as f:
            dict_calib = pickle.load(f) 
        print_file_datetime(pkl_calib)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), sharex=True, sharey=True, constrained_layout=True)
        if axs.ndim ==1 : axs=axs[:,np.newaxis]
        dict_corr={}
        vmax_corr = 0
        for i in range(2):
            for j,(c, conf) in enumerate(dict_conf.items()):
                hcalib = dict_calib[c]["h"]
                d_tomo = dict_out[c]
                htomo = d_tomo["h"]
                bx, by = d_tomo["binx"], d_tomo["biny"]
                ax = axs[i, j]
                if i == 0 : 
                    im = ax.pcolormesh(bx, by, htomo, norm=LogNorm(1, vmax))#, norm=LogNorm(1, vmax))
                else: 
                    htomo_corr = correct_tomo_images(htomo, hcalib)
                    vmax= np.nanmax(htomo_corr)
                    vmax_corr = vmax if vmax > vmax_corr  else vmax_corr
                    dict_corr[c] = {"h":htomo_corr,"binx":bx,"biny":by}
        norm_corr = LogNorm(vmin=1e-3, vmax=vmax_corr)
        for j, (c, corr) in enumerate(dict_corr.items()):
            h, bx, by = corr["h"], corr["binx"], corr["biny"]
            h = np.flipud(h) if (tel.name == "SXF") or (tel.name == "OM") else h
            ax = axs[1, j]
            im = ax.pcolormesh(bx, by, h, norm=norm_corr)#, norm=LogNorm(1, vmax))
            if j == len(dict_conf) - 1 : 
                cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-3,loc = 'right')
                cb= fig.colorbar(im, cax=cax, extend='max')
                cb.set_label(f'Entries Norm OS', labelpad=1)
                cb.ax.tick_params(which="both", labelsize="x-large",pad=1)    
        fout_png = cmn.plot_path / "images_corr.png"
        fig.savefig(fout_png)
        print(f"Save {fout_png}")
        fout_pkl = cmn.pkl_path / "images_corr.pkl"
        with open(fout_pkl, 'wb') as f:
            pickle.dump(dict_corr, f)
        print(f"Save {fout_pkl}")
    
    
    exit()



    ga = GeometricalAcceptance(tel)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = (ncols*6, 6*nrows), layout="tight", sharex=True, sharey=True)
    if axs.ndim == 1: axs =axs[np.newaxis].T    
    for i,(c, conf) in enumerate(dict_conf.items()):   
        mask = (df['config'] == c)
        x, y = df["dx_dz"][mask], df["dy_dz"][mask] 
        shp = tel.azimuth_matrix[c].shape
        bins = shp #
        h, binx, biny = np.histogram2d(x, y, bins=bins, range=conf.range_tanthetaxy)
        dz = conf.length_z
        acc = ga.angular_acceptance_map(u_edges=binx, v_edges=biny, delta_z=dz, Lx=800, Ly=800)
        ax = axs[0, i]
        im = ax.pcolormesh(binx, biny, acc, norm=LogNorm(1e-2, np.max(acc)))#, norm=LogNorm(1, vmax))
        h_corr = ga.correct_histogram(h, u_edges=binx, v_edges=biny, delta_z=dz, Lx=800, Ly=800)
        ax = axs[1, i]
        im = ax.pcolormesh(binx, biny, h_corr, cmap="coolwarm", norm=LogNorm(np.min(h_corr), np.max(h_corr)))#, norm=LogNorm(1, vmax))
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cb= fig.colorbar(im, cax=cax, extend='max')
        # cb.set_label(f'Acceptance', labelpad=5)

    file_out = dir_out / "geom_acc.png"
    fig.savefig(file_out)
    print(f"Save {file_out}")