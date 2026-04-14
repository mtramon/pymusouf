

from matplotlib.colors import LogNorm, Normalize
import numpy as np

def test_run_key(tel_name, fh5):
    for run_name in ["tomo", "tomo_combined"]:
        if tel_name in fh5 and run_name in fh5[tel_name]:
            return run_name
    raise KeyError(f"Neither 'tomo' nor 'tomo_combined' found for {tel_name}")

def set_norm(arr):
    vmin, vmax = np.nanmin(arr[arr!=0]), np.nanmax(arr)
    norm = LogNorm(vmin, vmax) if vmax/vmin > 2e1 else Normalize(vmin, vmax)
    return norm

