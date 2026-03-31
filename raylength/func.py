import numpy as np
from matplotlib.colors import LogNorm, Normalize
import h5py
from mpl_toolkits.axes_grid1 import inset_locator
import matplotlib.pyplot as plt

def save_rays_length_hdf5(h5file, tel_name, conf_name, rays_length):

    grp_tel = h5file.require_group(tel_name)
    grp = grp_tel.require_group(conf_name)

    if "rays_length" in grp:
        del grp["rays_length"]

    grp.create_dataset(
        "rays_length",
        data=rays_length,
        compression="gzip"
    )

def load_from_hdf5(h5file, tel_name, conf_name, key="rays_length"):
    grp = h5file[tel_name][conf_name]
    return np.array(grp[key])

def set_norm(arr):
    vmin, vmax = np.nanmin(arr[arr!=0]), np.nanmax(arr)
    norm = LogNorm(vmin, vmax) if vmax/vmin > 2e1 else Normalize(vmin, vmax)
    return norm

def plot_configuration(axs, h5file, tel, mask=None):
    configurations=tel.configurations.items()
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    assert len(axs) == len(configurations), "Axes shape does not match number of configurations"
    for j,(_, conf) in enumerate(configurations):
        ax = axs[j]
        array = load_from_hdf5(h5file, tel.name, conf.name)
        array = np.flipud(array) if (tel.name == "SXF") or (tel.name == "OM") else array.reshape(-1, order="C")
        u_edges, v_edges = conf.u_edges, conf.v_edges
        u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
        # m = mask[conf.name].reshape(conf.shape_uv) if mask is not None else np.ones(conf.shape_uv, dtype=bool)
        array = array.reshape(conf.shape_uv)
        # array[~m] = np.nan
        im = ax.pcolormesh(u,v, array.T)#norm=set_norm(array))
        cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-2,loc = 'right')
        cb = plt.colorbar(im, cax=cax, extend='max')
        ax.label_outer()
        ax.set_aspect('equal')