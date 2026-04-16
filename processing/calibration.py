import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from utils.functions import langau
from utils.tools import print_file_datetime, ask_yes_no


def load_adc_reference(filepath):
    """Load ADC reference values from an existing JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    return {
        int(k): {coord: params for coord, params in v.items()}
        for k, v in data.items()
    }


def adc_ref_all_none(ref):
    """Return True when the calibration result contains no usable values."""
    if not isinstance(ref, dict) or len(ref) == 0:
        return True

    for plane_values in ref.values():
        if isinstance(plane_values, dict):
            if any(value is not None for value in plane_values.values()):
                return False
        elif plane_values is not None:
            return False

    return True


def run_adc_calibration(tel, run, event_stream, n_planes, chunk_size=8000, bins=50, confirm_overwrite=ask_yes_no):
    """Run the full ADC calibration workflow and return the reference values."""
    calib = Calibration(n_planes=n_planes)
    fout_cal_json = run.dirs["json"] / "adc_ref.json"
    fout_cal_png = run.dirs["png"] / "adc_ref.png"

    print("\n=== ADC calibration ===")
    print("This step estimates the reference ADC values for each detector plane.")
    print_file_datetime(fout_cal_json)

    adc_ref_loaded = load_adc_reference(fout_cal_json)
    run_new_calibration = True

    if adc_ref_loaded is not None:
        print(f"An existing calibration file was found: {fout_cal_json}")
        if confirm_overwrite is None:
            run_new_calibration = False
        else:
            run_new_calibration = confirm_overwrite(
                "Do you want to run a new calibration and overwrite the existing file?",
                default="n",
            )

    if not run_new_calibration:
        print("Using the existing ADC reference values.")
        return adc_ref_loaded

    print("Running ADC calibration...")
    dict_ixpos = {p.id: p.position.index for p in tel.panels}
    chunks_calib = event_stream.chunked(chunk_size)
    adc_ref = calib.process_chunks_parallel(
        chunks_calib,
        dict_ixpos,
        n_jobs=None,
        bins=bins,
    )

    if adc_ref_all_none(adc_ref):
        print("No reliable ADC reference values were produced during calibration.")
        if adc_ref_loaded is not None:
            print("The previous ADC reference values will be kept.")
            return adc_ref_loaded

        print("Tracking will continue without ADC reference values.")
        return adc_ref

    calib.save_json(fout_cal_json)
    print(f"Saved ADC reference values to {fout_cal_json}")
    calib.save_distributions(fout_cal_png)
    print(f"Saved ADC distributions to {fout_cal_png}")
    return adc_ref


class Calibration:
    """
    Fit MIP-like event ('gold') ADC distributions to get reference values per plane and coordinate.
    """
    def __init__(self, n_planes: int):
        self.n_planes = n_planes
        self.adc_per_plane = {z: {"x": [], "y": []} for z in range(n_planes)}
        self.adc_mpv = {z: {"x": None, "y": None} for z in range(n_planes)}
        self.adc_ref = {z: {"x": None, "y": None} for z in range(n_planes)}
        self.threshold = None

    def process_chunk(self, chunk, dict_ixpos):
        adc_per_plane = {z: {"x": [], "y": []} for z in range(self.n_planes)}
        for evt in chunk:
            evt.get_xyz(dict_ixpos=dict_ixpos)
            if len(evt.xyz) < self.n_planes:
                continue
            z_vals = evt.xyz[:, 2].astype(int)
            multiplicity = np.bincount(z_vals, minlength=self.n_planes)
            if not np.all(multiplicity == 1):
                continue
            for z in range(self.n_planes):
                idx = np.where(z_vals == z)[0][0]
                adc_per_plane[z]["x"].append(float(evt.adc[idx, 0]))
                adc_per_plane[z]["y"].append(float(evt.adc[idx, 1]))
        return adc_per_plane

    def _find_separation_threshold(self, values, bins=50, low_adc_max=300, min_valley_ratio=0.3):
        """
        Détecte un pic SPE bas et un pic MIP haut.
        Gère le cas où le pic SPE est dans le premier bin.
        Retourne le seuil (float) ou None si pas de séparation claire.
        """
        values = np.asarray(values)
        if len(values) < 100:
            return None

        hist, edges = np.histogram(values, bins=bins)
        x = 0.5 * (edges[:-1] + edges[1:])
        
        # Étendre l'histogramme avec un bin à gauche (valeur 0)
        hist_ext = np.concatenate(([0], hist))
        x_ext = np.concatenate(([x[0] - (x[1]-x[0])], x))  # bin fictif
        
        max_h = np.max(hist_ext)
        if max_h == 0:
            return None
        
        # Détection sur histogramme étendu
        prominence = max(2, 0.05 * max_h)
        peaks, props = find_peaks(hist_ext, prominence=prominence, distance=1)
        # print(f"Found peaks with set prominence 0.05 * {max_h} at {x_ext[peaks]} with heights {hist_ext[peaks]} and prominences {props['prominences']}")
        if len(peaks) < 2:
            prominence = max(1, 0.02 * max_h)
            peaks, props = find_peaks(hist_ext, prominence=prominence, distance=1)
            # print(f"Found peaks with set prominence 0.02 * {max_h} at {x_ext[peaks]} with heights {hist_ext[peaks]} and prominences {props['prominences']}")
        # Convertir les indices en indices réels (enlever le bin factice)
        real_peaks = [p-1 for p in peaks if p-1 >= 0 and p-1 < len(hist)]
        # Cas particulier : premier bin réel très haut (pic SPE) mais non détecté
        if len(real_peaks) < 2 and hist[0] > 0.5 * max_h and hist[0] > np.median(hist):
            # Chercher le pic MIP dans les bins suivants (le plus haut après le premier)
            if len(hist) > 1:
                mip_idx = np.argmax(hist[1:]) + 1
                if hist[mip_idx] > 0:
                    real_peaks = [0, mip_idx]
                else:
                    return None
            else:
                return None
        if len(real_peaks) < 2:
            return None
        
        # Prendre les deux pics les plus hauts
        heights = hist[real_peaks]
        sorted_idx = np.argsort(heights)[::-1]
        highest_two = [real_peaks[i] for i in sorted_idx[:2]]
        peaks_sorted = np.sort(highest_two)
        first_peak, second_peak = peaks_sorted[0], peaks_sorted[1]
        
        # Vérifier que le premier pic est bas (SPE)
        adc_first = x[first_peak]
        if adc_first > low_adc_max:
            return None
        # Vérifier que le second pic est plus haut en ADC
        adc_second = x[second_peak]
        if adc_second <= adc_first:
            return None
        
        # Profondeur de la vallée entre les deux pics
        valley_region = hist[first_peak:second_peak+1]
        valley_min = np.min(valley_region)
        if valley_min > (1 - min_valley_ratio) * hist[second_peak]:
            print(f"Valley too shallow: min {valley_min} > {(1 - min_valley_ratio) * hist[second_peak]}")
            return None
        
        valley_idx = np.argmin(valley_region) + first_peak
        threshold = x[valley_idx]
        
        # Sécurité : garder au moins 20% des événements
        if np.sum(values > threshold) < 0.2 * len(values):
            print("np.sum(values > threshold) < 0.2 * len(values)")
            return None
        self.threshold = threshold
        return threshold
    
    # def remove_spe_peak(self, values, bins):
    #     """
    #     Supprime le pic SPE uniquement s'il est clairement identifié.
    #     Sinon, retourne les valeurs inchangées.
    #     """
    #     values = np.asarray(values)
    #     if len(values) < 100:
    #         return values
        
    #     threshold = self._find_separation_threshold(values, bins=bins)
    #     print(f"Identified threshold for SPE separation: {threshold}")
    #     if threshold is not None:
    #         filtered = values[values > threshold]
    #         # On vérifie qu'il reste assez d'événements (sinon on garde tout)
    #         if len(filtered) >= 0.5 * len(values):
    #             return filtered
    #     # Aucune coupure si pas de seuil fiable
    #     return values
    

    def remove_spe_peak(self, adc_values, bins=50):
        values = np.asarray(adc_values)
        if len(values) < 50:
            return values
        hist, edges = np.histogram(values, bins=bins)
        x = 0.5 * (edges[:-1] + edges[1:])
        # --- lissage léger ---
        hist_smooth = np.convolve(hist, np.ones(3)/3, mode="same")
        # --- pic principal (souvent SPE) ---
        i_peak = np.argmax(hist_smooth)
        # --- recherche vallée après le pic ---
        valley_idx = None
        for i in range(i_peak + 1, len(hist_smooth) - 1):
            if hist_smooth[i] < hist_smooth[i-1] and hist_smooth[i] < hist_smooth[i+1]:
                valley_idx = i
                break
        # --- cas normal ---
        if valley_idx is not None:
            threshold = x[valley_idx]
        else:
            # --- fallback adaptatif ---
            # distribution monotone → SPE dominant
            p50 = np.percentile(values, 50)
            p20 = np.percentile(values, 20)
            # si distribution très asymétrique → couper bas ADC
            if p50 < 2 * p20:
                threshold = np.percentile(values, 30)
            else:
                threshold = np.percentile(values, 20)
        # --- protection ---
        threshold = max(threshold, np.percentile(values, 10))
        filtered = values[values > threshold]
        # --- sécurité : éviter échec silencieux ---
        if len(filtered) < 0.2 * len(values):
            return values  # on abandonne la coupe
        return filtered

    def _fallback_mpv_sigma(self, values, bins):
        values = np.asarray(values)
        if len(values) == 0:
            return None
        hist, edges = np.histogram(values, bins=bins)
        imax = np.argmax(hist)
        mpv = 0.5 * (edges[imax] + edges[imax + 1])
        # sigma robuste (corrigé)
        p16 = np.percentile(values, 16)
        p84 = np.percentile(values, 84)
        sigma = 0.5 * (p84 - p16)
        return {
            "mpv": float(mpv),
            "mpv_raw": float(mpv),
            "eta": None,
            "sigma": float(sigma),
            "A": int(hist.max()),
            "method": "fallback"
        }
    
    def fit_langau(self, adc_values, bins):
        adc_values = np.asarray(adc_values)
        nentries = len(adc_values)
        if nentries < 30:
            return self._fallback_mpv_sigma(adc_values, bins)
        # --- SPE filter ---
        adc_values = self.remove_spe_peak(adc_values, bins=bins)
        if nentries < 30:
            return self._fallback_mpv_sigma(adc_values, bins)
        # --- histogram ---
        hist, edges = np.histogram(adc_values, bins=bins)
        x = 0.5 * (edges[:-1] + edges[1:])
        # --- init robuste ---
        try:
            mpv0 = x[np.argmax(hist)]
            # eta0 = np.std(adc_values) / 2.0 
            # sigma0 = eta0 / 2.0
            sigma_est = 0.5 * (np.percentile(adc_values, 84) - np.percentile(adc_values, 16))
            eta0 = sigma_est
            sigma0 = sigma_est / 2
            A0 = hist.max()
            p0 = [mpv0, eta0, sigma0, A0]
            bounds = (
                [0, 1e-3, 1e-3, 0],
                [10000, 1000, 1000, 1e9]
            )
            popt, _ = curve_fit(
                langau,
                x,
                hist,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            if (popt[0] / popt[2] > 6) or (popt[0] / popt[2] < 1):
                print(f"Unreliable fit: MPV/sigma ratio = {popt[0] / popt[2]}")
                raise RuntimeError()
        except Exception:
            return self._fallback_mpv_sigma(adc_values, bins)
        mpv_raw, eta, sigma, A = popt
        # --- MPV réel ---
        try:
            yfit = langau(x, *popt)
            mpv = x[np.argmax(yfit)]
        except Exception:
            return self._fallback_mpv_sigma(adc_values, bins)

        # --- sanity check ---
        if not np.isfinite(mpv) or sigma <= 0 or eta <= 0:
            return self._fallback_mpv_sigma(adc_values, bins)

        return {
            "mpv": float(mpv),
            "mpv_raw": float(mpv_raw),
            "eta": float(eta),
            "sigma": float(sigma),
            "A": float(A),
            "nentries": int(nentries),
            "method": "fit"
        }


    def compute_langau(self, bins):
        adc_ref = {}
        for z, coord_dict in self.adc_per_plane.items():
            adc_ref[z] = {"x": None, "y": None}
            for coord in ("x", "y"):
                values = coord_dict[coord]
                print(
                    f"Plane {z} coord {coord} - "
                    f"{len(values)} entries"
                )
                if len(values) == 0:
                    adc_ref[z][coord] = None
                    continue

                if len(values) < 100:
                    res = self._fallback_mpv_sigma(values, bins)
                    adc_ref[z][coord] = res
                    print(f"→ fallback (low stats)")
                    continue

                res = self.fit_langau(values, bins=bins)

                if res is None:
                    res = self._fallback_mpv_sigma(values, bins)
                    print(f"→ fallback (fit failed)")
                else:
                    print(f"→ fit ok ({res['method']})")
                
                adc_ref[z][coord] = res
            
            print()
        self.adc_ref = adc_ref
        return self.adc_ref

    def merge_adc_dicts(self, list_of_dicts):
        merged = {z: {"x": [], "y": []} for z in range(self.n_planes)}
        for d in list_of_dicts:
            for z in range(self.n_planes):
                merged[z]["x"].extend(d[z]["x"])
                merged[z]["y"].extend(d[z]["y"])
        return merged

    def process_chunks_parallel(self, chunks, dict_ixpos, n_jobs=None, bins=100):
        if n_jobs is None:
            n_jobs = cpu_count() - 1
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
        )(
            delayed(self.process_chunk)(chunk, dict_ixpos)
            for chunk in chunks
        )
        self.adc_per_plane = self.merge_adc_dicts(results)
        self.compute_langau(bins=bins)
        return self.adc_ref

    def save_json(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(str(filepath), "w") as f:
            json.dump(self.adc_ref, f, indent=2)

    def save_distributions(self, filepath, bins=50):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(
            nrows=2,
            ncols=self.n_planes,
            figsize=(self.n_planes * 4, 8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = np.array(axs)
        if axs.ndim == 1:
            axs = axs.reshape(2, self.n_planes)

        for z, coord_dict in self.adc_per_plane.items():
            for row, coord, color in ((0, "x", "tab:blue"), (1, "y", "tab:orange")):
                ax = axs[row, z]
                values = coord_dict[coord]
                if len(values) == 0:
                    ax.set_visible(False)
                    continue
                hist, edges, _ = ax.hist(
                    values,
                    bins=bins,
                    alpha=0.7,
                    label=f"{coord.upper()} data",
                    color=color,
                )
                if self.adc_ref.get(z) and self.adc_ref[z].get(coord) and self.adc_ref[z][coord].get("method")== "fit":
                    params = self.adc_ref[z][coord]
                    x_fit = np.linspace(edges[0], edges[-1], 1000)
                    y_fit = langau(x_fit, params["mpv_raw"], params["eta"], params["sigma"], params["A"])
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=color,
                        linestyle="-",
                        label=f"{coord.upper()} fit \nMPV={params['mpv']:.1f}, \n$\\sigma$={params['sigma']:.1f}, \n$\\eta$={params['eta']:.1f}, \nA={params['A']:.0f}",
                    )
                elif self.adc_ref.get(z) and self.adc_ref[z].get(coord):
                    params = self.adc_ref[z][coord]
                    ax.axvline(
                        params["mpv"],
                        color=color,
                        linestyle="-",
                        label=f"{coord.upper()} fallback \nMPV={params['mpv']:.1f}, \n$\\sigma$={params['sigma']:.1f}",
                    )
                else: pass
                
                threshold = self._find_separation_threshold(values)
                if threshold is not None:
                    ax.axvline(
                        threshold,
                        color=color,
                        linestyle="dashed",
                        alpha=0.5,
                        linewidth=2,
                        label=f"Threshold SPE: {threshold:.1f} ADC"
                    )
                
                ax.set_xlabel("ADC")
                ax.set_ylabel("Counts")
                title_coord = "X" if row == 0 else "Y"
                ax.set_title(f"Plane {z} - {title_coord}")
                ax.label_outer()
                ax.legend(fontsize="small")

        fig.savefig(filepath, dpi=120)
        plt.close(fig)