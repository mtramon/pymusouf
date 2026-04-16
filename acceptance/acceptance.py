import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.axes_grid1 import inset_locator

from telescope import tel_SNJ

params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':"xx-large",
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


def acceptance_old(
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

def geometrical_acceptance (u_edges, v_edges, delta_z=120,Lx=80,Ly=80,nu_sr=5, nv_sr=5):
    nu_pix = len(u_edges) - 1
    nv_pix = len(v_edges) - 1
    res = np.zeros((nu_pix, nv_pix))
    for j in range(nv_pix):
        for i in range(nu_pix):
            u0, u1 = u_edges[i], u_edges[i+1]
            v0, v1 = v_edges[j], v_edges[j+1]
            du = (u1 - u0) / nu_sr
            dv = (v1 - v0) / nv_sr
            # sous-bins réguliers (midpoint rule)
            u_sr = u0 + (np.arange(nu_sr) + 0.5) * du
            v_sr = v0 + (np.arange(nv_sr) + 0.5) * dv
            U, V = np.meshgrid(u_sr, v_sr, indexing='ij')
            S = np.maximum(0, Lx - abs(U)*delta_z) * np.maximum(0, Ly - abs(V)*delta_z)
            dG = S * du * dv / (1 + U**2 + V**2)**2
            res[i,j] = dG.sum()
    return res


if __name__ == "__main__":

    conf_name="4p"
    conf = tel_SNJ.configurations[conf_name]
    u_edges, v_edges = conf.u_edges, conf.v_edges
    u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
    res = geometrical_acceptance(u_edges, v_edges, delta_z=conf.length_z*1e-1)
    print(res.shape)
    fig, ax = plt.subplots(nrows=1)
    im = ax.pcolormesh(u,v,res)
    cax = inset_locator.inset_axes(ax   , width="4%",  height="100%", borderpad=-2,loc = 'right')
    cb= fig.colorbar(im, cax=cax, extend='max')
    cb.set_label(f"Acceptance [cm$^2$.sr]" , labelpad=1)
    print(res.min(), res.max())
    fout = "test.png"
    fig.savefig(fout)
    print(f"Save {fout}")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
