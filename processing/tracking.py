#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from itertools import product
from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from skimage.measure import ransac, LineModelND
from sklearn.linear_model import RANSACRegressor,LinearRegression
import sys
import time
#package module(s)
from data import RawData
from telescope import ChannelMap, Telescope
from utils.tools import format_time

@dataclass(slots=True)
class Hit:
    channel_no : int
    bar_no : int
    adc : float
    panel_id : int = field(default=0)

@dataclass(slots=True)
class Timestamp: 
    s : int #sec
    ns : int  #nanosec
    res: float = field(default=1e-8)  # explicite

@dataclass(slots=True)
class ImpactPanel:
    """XY hits collection on panel"""
    zpos : float
    panel_id :int= field(default=None)
    hits : List[Hit] = field(default_factory=list)


@dataclass(slots=True)
class ImpactPM: 
    """Hit collections on one PMT
    Useful for configuration where one PMT is connected to two detection panels"""
    line : str
    evt_id : int = field(init=False)
    timestamp : Timestamp = field(init=False)
    pm_id: int = field(init=False)
    nhits: int = field(init=False)
    panel_impacts : Dict[int, ImpactPanel] = field(default_factory=dict)
    _hits_raw : list = field(init=False)
    
    def __post_init__(self):
        self.readline()

    def readline(self):
        l = self.line.split()
        if len(l) < 9:
            raise ValueError(f"Malformed line: {self.line}")
        ts_s = float(l[0])
        if ts_s < 1e9:
            raise ValueError(f"Inconsistent timestamp: {self.line}")
        self.evt_id = int(l[1])
        ts_ns = float(l[2])
        self.pm_id = int(l[5])
        self.nhits = int(l[8])
        expected_len = 9 + 2 * self.nhits
        if len(l) < expected_len:
            raise ValueError(f"Inconsistent nhits: {self.line}")
        self.timestamp = Timestamp(s=ts_s, ns=ts_ns)
        hits = l[9:expected_len]
        # self._channel_no = list(map(int, hits[0::2]))
        # self._adc = list(map(float, hits[1::2]))
        self._hits_raw = [(int(ch), float(adc)) for ch, adc in zip(hits[0::2], hits[1::2])]

    def compute_panel_id(self, ch, npm, minPlan, maxPlan):
        if npm == 2:
            side = 0 if ch % 8 <= 3 else 1
            base = self.pm_id - minPlan if maxPlan is None else maxPlan - self.pm_id
            return 2 * base + side
        elif npm in (3, 4):
            return self.pm_id
        else:
            raise ValueError("Unknown PMT configuration")

    def fill_panel_impacts(self,channelmap:ChannelMap, npm:int, zpos:dict, minPlan:int=6, maxPlan:int=None, scint_eff:float=None):
        chmap  = channelmap.dict_ch_to_bar
        # rand = np.random.random(len(self._adc))
        for i, (ch, adc) in enumerate(self._hits_raw):
            if ch not in chmap:
                continue
            # if scint_eff is not None and rand[i] > scint_eff:
            #     continue
            pan_id = self.compute_panel_id(ch, npm, minPlan, maxPlan)
            if pan_id not in self.panel_impacts.keys():
                try:
                    z = zpos[pan_id]
                except KeyError:
                    raise KeyError(f"Missing zpos for panel {pan_id}")
                self.panel_impacts[pan_id] = ImpactPanel(
                    panel_id=pan_id,
                    zpos=zpos[pan_id]
                )
            self.panel_impacts[pan_id].hits.append(
                Hit(channel_no=ch, bar_no=chmap[ch], adc=adc, panel_id=pan_id)
            )

def convert_bar_to_mm(xyz: np.ndarray, scint_width:np.ndarray, z_mm:np.ndarray, norm_factor=None) -> np.ndarray:
    # Séparation des colonnes
    x_bar = xyz[:, 0]
    y_bar = xyz[:, 1]
    p_ix  = xyz[:, 2].astype(int) 
    w = scint_width[p_ix]
    z_new = z_mm[p_ix]
    # Facteurs multiplicatifs
    scale = np.column_stack((w, w, z_new))
    # Coordonnées centrées
    base = np.column_stack((x_bar - 0.5, y_bar - 0.5, np.ones_like(x_bar)))
    if norm_factor is not None:
        base[:,:2] *= norm_factor[:,np.newaxis]
    return base * scale

def normalize_xy_coords(xyz: np.ndarray, pitch_xy, pitch_ref):
    x, y, z = xyz.T
    norm_factor =  pitch_ref / pitch_xy[z.astype(int)]
    return np.array([x / norm_factor, y / norm_factor, z], dtype=float).T, norm_factor

 # --------------------------------------------------
# A tester: Jitter contrôlé uniforme dans le barreau
# --------------------------------------------------
def apply_jitter(hits, scale=0.5):
    jitter = np.random.uniform(
        -scale,
        scale,
        size=hits.shape
    )
    jitter[:, 2:] = 0  # jamais de jitter en Z
    return hits + jitter


@dataclass(slots=True)
class Event:
    id : int 
    timestamp : Timestamp 
    tof : float = field(init=False) #time-of-flight
    xyz : np.ndarray = field(init=False)
    npts : int = 0
    adc : np.ndarray = field(init=False)
    nimpacts : int = field(init=False) 
    impacts : Dict[int, ImpactPanel] = field(default_factory=dict) 
    gold : bool = 0 #if an event forms exactly 1 hit per scintillation layer

    def _xyz_bar(self, dict_ixpos) -> np.ndarray:
        bar_no = []
        adc = []
        for _, imp in self.impacts.items():  # loop on impacts (=impacted panels)
            p_ix = dict_ixpos[imp.panel_id]
            hits_x = [h for h in imp.hits if isinstance(h.bar_no, str) and h.bar_no[0] == 'X']
            hits_y = [h for h in imp.hits if isinstance(h.bar_no, str) and h.bar_no[0] == 'Y']
            if not hits_x or not hits_y:
                continue
            for hx in hits_x:
                for hy in hits_y:
                    bar_no.append((int(hx.bar_no[1:]), int(hy.bar_no[1:]), p_ix))
                    adc.append((hx.adc, hy.adc, p_ix))
        arr_xyz_bar = np.asarray(bar_no)
        self.adc = np.asarray(adc)
        return arr_xyz_bar
        
    def get_xyz(self, dict_ixpos:dict, mm:bool=False, scint_width=None, z_mm=None) -> None:
        """
    
        """
        self.xyz = self._xyz_bar(dict_ixpos)
        if len(self.xyz) == 0: return 
        if mm: self.xyz = convert_bar_to_mm(self.xyz, scint_width, z_mm)
        self.npts = self.xyz.shape[0]
        self.nimpacts = len(list(set(self.xyz[:,-1]))) 

    def get_time_of_flight(self) -> None:
        l_impacts= list(self.impacts.values())
        l_z = [imp.zpos for imp in l_impacts]
        imp_front, imp_rear =  l_impacts[np.argmin(l_z)], l_impacts[np.argmax(l_z)]
        tns_front, tns_rear = imp_front.timestamp.ns, imp_rear.timestamp.ns
        self.tof = (tns_rear-tns_front) #in 10ns

class TrackModel:
    def __init__(self,  min_planes:int=3):
        self.min_planes = min_planes

    def _select_one_hit_per_plane(self, xyz):
        """
        Garde au plus un hit par plan (le plus proche de la droite brute)
        """
        out = []
        for z in np.unique(xyz[:, 2]):
            hits = xyz[xyz[:, 2] == z]
            if hits.shape[0] == 1:
                out.append(hits[0])
            else:
                # distance à une droite verticale approximative
                bary = hits[:, 0:2].mean(axis=0)
                d = np.linalg.norm(hits[:, 0:2] - bary, axis=1)
                out.append(hits[np.argmin(d)])
        return np.asarray(out)

class RansacModel(TrackModel):
        
    def __init__(
        self,
        residual_threshold=1,
        min_samples=2, 
        max_trials=40,
        min_planes=3,
    ):
        """
        """
        TrackModel.__init__(self, min_planes=min_planes)
        ###Ransac params
        self.residual_threshold = residual_threshold
        self.min_samples = min_samples
        self.max_trials = max_trials
        self.min_planes = min_planes
        ###
        self.model = None
        self.inliers = None
        # self.tan_max = tan_max
        self.valid = False

    def fit(self, xyz):
        if xyz.shape[0] < self.min_planes:
            return
        try:
            model_ransac, inliers = ransac(
                xyz,
                LineModelND,
                min_samples=self.min_samples,
                residual_threshold=self.residual_threshold,
                max_trials=self.max_trials
            )
        except ValueError:
            return
        if inliers.sum() < self.min_planes:
            return
        xyz_in = xyz[inliers]
        # xyz_in = self._select_one_hit_per_plane(xyz_in)
        if xyz_in.shape[0] < self.min_planes:
            return
        # --- RMS
        rms = np.sqrt(np.mean(model_ransac.residuals(xyz_in)**2))
        # xyz_in_front = xyz_in[np.argmin(xyz_in[:, 2])]
        # xyz_in_rear = xyz_in[np.argmax(xyz_in[:, 2])]
        # rms = np.sqrt(np.mean(model_ransac.residuals(np.vstack((xyz_in_front, xyz_in_rear)))**2))
        model = dict(
            origin=model_ransac.origin,
            direction=model_ransac.direction,
            inliers_idx=inliers,
            rms=rms,
            n_planes=xyz_in.shape[0],
        )
        return model

class DeterministicModel:
    """
    Tracker déterministe pour détecteur à plans XY discrets.
    Entrée : array xyz de forme (n,3) en indices de barreaux.
    """
    def __init__(
        self,
        nx=[16]*4,
        ny=[16]*4,
        residual_threshold=0.75,
        max_rms=0.6,
        min_planes=3,
    ):
        TrackModel.__init__(self, min_planes=min_planes)
        self.nx = nx
        self.ny = ny
        self.residual_threshold = residual_threshold
        self.max_rms = max_rms
   
    # --------------------------------------------------
    
    # --------------------------------------------------
    # A tester: Barycentre pondéré
    # --------------------------------------------------
    def weighted_hit(self, hits, p0, v):
        pts = hits[:, :3]
        adc = hits[:, 3]
        # distance vectorisée
        diff = pts - p0
        cross = np.cross(diff, v)
        dists = np.linalg.norm(cross, axis=1)
        # sélection rapide
        idx = np.argsort(dists)
        idx = idx[:3]  # top 3 hits
        dsel = dists[idx]
        psel = pts[idx]
        adc_sel = adc[idx]
        # rejet rapide
        if dsel[0] > self.residual_threshold:
            return None, None
        # cas trivial (le plus fréquent)
        if len(idx) == 1 or dsel[1] > self.residual_threshold:
            return psel[0], dsel[0]
        # poids rapides
        sigma = 0.3
        w_geom = 1.0 / (1.0 + (dsel / sigma)**2)
        adc_norm = adc_sel / (np.mean(adc_sel) + 1e-6)
        w = w_geom * adc_norm
        bary = np.average(psel, axis=0, weights=w)
        rms = np.sqrt(np.average(dsel**2, weights=w))
        return bary, rms


    # --------------------------------------------------
    # Géométrie de base
    # --------------------------------------------------
    @staticmethod
    def _line_from_points(p1, p2):
        v = p2 - p1
        norm = np.linalg.norm(v)
        if norm == 0:
            return None, None
        return p1, v / norm

    @staticmethod
    def _point_line_distance(p, p0, v):
        return np.linalg.norm(np.cross(p - p0, v))

    @staticmethod
    def _intersect_z(p0, v, z):
        if abs(v[2]) < 1e-6:
            return None
        t = (z - p0[2]) / v[2]
        return p0 + t * v
    
    def _inside_active_area(self, x, y, z):
        # return (-0.5 <= x < self.nx - 0.5) and (-0.5 <= y < self.ny - 0.5)
        return (0.5 <= x < self.nx[int(z)] + 0.5) and (0.5 <= y < self.ny[int(z)] + 0.5)
    
    # --------------------------------------------------
    # Reconstruction principale
    # --------------------------------------------------
    def fit(self, hits):
        """
        hits : ndarray (n,4) avec colonnes (x_idx, y_idx, z_idx, adc_idx)

        Retourne :
          dict avec origin, direction, inliers, rms, n_planes
          ou None
        """
        if hits.shape[0] < self.min_planes:
            return None
        planes = np.unique(hits[:, 2].astype(int))
        if planes.size < self.min_planes:
            return None
                
        z_min = planes.min()
        z_max = planes.max()
        hits_min = hits[hits[:, 2] == z_min]
        hits_max = hits[hits[:, 2] == z_max]
        best = None
        
        for h1, h2 in product(hits_min, hits_max):
            p0, v = self._line_from_points(h1[:3], h2[:3])
            if p0 is None:
                continue
            used_planes = set()
            inliers_idx = []
            residuals = []
            
            for z in planes:
                ip = self._intersect_z(p0, v, z)
                if ip is None:
                    continue
                if not self._inside_active_area(ip[0], ip[1], ip[2]):
                    continue

                plane_hits = hits[hits[:, 2] == z]

                # bary, rms = self.weighted_hit(plane_hits, p0, v)
                # if bary is None:
                #     continue
                # dists = np.linalg.norm(
                #     abs(plane_hits[:, :3] - bary),
                #     axis=1
                # )
                # imin = np.argmin(dists)
                # inliers_idx.append(imin)
                # residuals.append(rms)
                # used_planes.add(z)

                dists = np.linalg.norm(
                    np.cross(plane_hits[:,:3] - p0, v),
                    axis=1
                )
                imin = np.argmin(dists)
                dmin = dists[imin]
                if dmin < self.residual_threshold:
                    # inliers.append(plane_hits[imin])
                    inliers_idx.append(imin)
                    residuals.append(dmin)
                    used_planes.add(z)
                    

            n_planes = len(used_planes)
            if n_planes < self.min_planes:
                continue
            rms_total = np.sqrt(np.mean(np.square(residuals)))
            if rms_total > self.max_rms:
                continue
            score = (n_planes, -rms_total)
            if best is None or score > best["score"]:
                best = dict(
                    origin=p0,
                    direction=v,
                    inliers_idx=np.array(inliers_idx),
                    rms=rms_total,
                    n_planes=n_planes,
                    score=score
                )
            # arrêt anticipé : modèle quasi parfait
            if n_planes == planes.size and rms_total < 0.3:
                break
        return best


class EventStream:
   
    def __init__(self, telescope: Telescope, data: RawData, entry_start: int = 0, nev_max: int = int(1e8)):
        self.tel = telescope
        self.data = data
        self.entry_start = entry_start
        self.nev_max = nev_max
        self.maxPlan = None if telescope.name != "SXF" else 7
        self.nev_tot = 0
        self.pmts = {pm.id: pm for pm in self.tel.pmts}
        self.dict_id_zpos = {p.id: p.position.z for p in self.tel.panels}

    def event_stream(self):
        npm = len(self.tel.pmts)
        minPlan = min(pm.id for pm in self.tel.pmts)
        evt = None
        last_evt_id = None

        for file in self.data.dataset:
            if self.nev_tot >= self.nev_max:
                break
            for line in self.data.readfile(file):
                if self.nev_tot >= self.nev_max:
                    break
                try:
                    impm = ImpactPM(line=line)
                except ValueError:
                    continue
                pmt = self.pmts.get(impm.pm_id)
                if pmt is None:
                    continue
                impm.fill_panel_impacts(
                    pmt.channelmap,
                    npm,
                    self.dict_id_zpos,
                    minPlan=minPlan,
                    maxPlan=self.maxPlan,
                )
                if evt is None or impm.evt_id != last_evt_id:
                    if evt is not None:
                        self.nev_tot += 1
                        yield evt
                    evt = Event(id=impm.evt_id, timestamp=impm.timestamp)
                    last_evt_id = impm.evt_id
                evt.impacts.update(impm.panel_impacts)

        if evt is not None and self.nev_tot < self.nev_max:
            self.nev_tot += 1
            yield evt

    def chunked(self, chunk_size: int = 2000):
        chunk = []
        for evt in self.event_stream():
            chunk.append(evt)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


    
class Tracking: 
   
    def __init__(self, telescope:Telescope, tan_max:float=1.):
        self.tel = telescope
        self.tan_max = tan_max
        self.panels = { p.id : p for p in self.tel.panels}
        self.dict_id_zpos = { id : p.position.z  for id,p in self.panels.items()}
        self.dict_ixpos_zpos = { p.position.index : p.position.z  for id,p in self.panels.items()}
        self.zpos = list(self.dict_id_zpos.values())
        self.dict_id_ixpos = { id : p.position.index  for id,p in self.panels.items()}
        self.dict_cfg_zpos = { cfg_name : [p.position.z  for p in cfg.panels] for cfg_name, cfg in self.tel.configurations.items()}
        self.dict_cfg_ix = { cfg_name : [p.position.index  for p in cfg.panels] for cfg_name, cfg in self.tel.configurations.items()}
        self.dict_ixpos_width = { p.position.index :  float(p.matrix.scintillator.width) for _,p  in self.panels.items() }
        self.arr_scint_width = np.array([self.dict_ixpos_width[i] for i in range(len(self.dict_ixpos_width))])
        self.arr_z_mm = np.array([self.dict_ixpos_zpos[i] for i in range(len(self.dict_ixpos_zpos))])
        self.last_evt_id = None
        self.pitch_xy = np.array([p.matrix.scintillator.width for _,p in self.panels.items()])
        self.pitch_ref = self.tel.panels[0].matrix.scintillator.width  

    def prefilter(self, evt:Event)->bool:
        npanels = len(self.panels)
        if len(evt.xyz) ==0 : return False
        #Capture selected events to be reconstructed
        if  evt.nimpacts  < npanels-1 : return False
        #check hit multiplicity on each impact
        max_multiplicity = max([len(i.hits) for _,i in evt.impacts.items()])
        if max_multiplicity > 10: return False
        return True

    def intersections(self, p0, v, cfg_name, eps=1e-12):
        """
        Intersections droite / plans z
        """
        points = []
        for z in self.dict_cfg_zpos[cfg_name]:
            t = (z - p0[2]) / (v[2]+eps)
            points.append(p0 + t * v)
        return np.asarray(points)


class RansacTracking(Tracking): 

    def __init__(self, telescope:Telescope, adc_ref:dict=None, ndisplays:int=0):
        Tracking.__init__(self, telescope)     
        self.maxPlan = None if telescope.name != "SXF" else 7
        self.ndisplays = ndisplays
        self.adc_ref = adc_ref
        # self.adc_mpv = {cfg: np.array([self.adc_ref[i]["mpv"] for i in z_ix])for cfg, z_ix in self.dict_cfg_ix.items()}
        # self.adc_sigma = {cfg: np.array([self.adc_ref[i]["sigma"] for i in z_ix]) for cfg, z_ix in self.dict_cfg_ix.items()}
        # print(self.adc_mpv, self.adc_sigma)

    def main(self, file_out=None, chunks=None, n_jobs=None, **kwargs_model):
        file_out = Path(file_out)
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
        # config légère à passer aux workers
        config = {
            "tracker_class": self.__class__,
            "init": {
                "telescope": self.tel,
                "adc_ref": self.adc_ref,
            },
            "ndisplays": self.ndisplays,
        }
        dicts = {
            "ixpos": self.dict_id_ixpos,
            "zpos": self.dict_id_zpos
        }
        thresh = kwargs_model["residual_threshold"]
        nx, ny = [],[]
        for p in self.tel.panels :  nx.append(p.matrix.nbars_x), ny.append(p.matrix.nbars_y)
        det_config = dict(
                nx=nx,
                ny=ny,
                residual_threshold=thresh
            )
        ransac_config = kwargs_model
        models_config = {"det": det_config, "ransac":ransac_config }
        
        if chunks is None:
            raise ValueError("Event chunks generator is required for RansacTracking.main()")
        start_time = time.time()
        total_events = 0
        total_processed = 0
        total_tracked = 0
        last_print = 0

        results_tracking = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator"
        )(
            delayed(_tracking_chunk_static)(chunk, config, dicts, models_config)
            for chunk in chunks
        )
        if file_out.exists():
            os.remove(file_out)
            print(f"Removed old {file_out}\n")

        columns = ["event_id", "timestamp", 
                   "nhits_0","nhits_1","nhits_2","nhits_3",
                   "config",
                   "dx_dz","dy_dz","dz",
                   "rms",
                   "ransac",
                   "mip_score_x","mip_score_y"]
        pd.DataFrame(columns=columns).to_csv(
            file_out,
            index=False,
            compression="gzip"
        )
        # ----------------------------------
        # EVENT DISPLAY
        # ----------------------------------        
        plots_dict = {}      #  {config: data}
        n_displayed =    0
        for rows, plots, ntot, nprocess, ntrack in results_tracking:
            total_events += ntot
            total_processed += nprocess
            total_tracked += ntrack
            # écriture
            if rows:
                df = pd.DataFrame(rows, columns=columns)
                df.to_csv(
                        file_out,
                        mode='a',
                        header=False,
                        index=False,
                        compression="gzip"
                    )
            # display
            if len(plots) > 0:
                for evt_id, cfg, xyz, weights, inliers, adc, p0, v, rms, use_ransac, mip_score in plots:
                    if evt_id not in plots_dict:
                        if n_displayed >= self.ndisplays: break
                        plots_dict[evt_id] = {
                            cfg: (xyz, weights, inliers, adc, p0, v, rms, use_ransac, mip_score)
                        }
                        n_displayed += 1
                    else:
                        plots_dict[evt_id][cfg] = (xyz, weights, inliers, adc, p0, v, rms, use_ransac, mip_score)

            # ----------------------------------
            # AFFICHAGE TEMPS REEL
            # ----------------------------------
            now = time.time()
            # limiter fréquence affichage
            if now - last_print > 0.5:
                elapsed = now - start_time
                elapsed_str = format_time(elapsed)
                rate = total_processed / elapsed if elapsed > 0 else 0
                if total_events > 0:
                    remaining = total_events - total_processed
                    eta = remaining / rate if rate > 0 else 0
                    eta_str = format_time(eta)
                    msg = (
                        f"{total_processed}/{total_events} | "
                        f"{rate:,.0f} evt/s | "
                        f"ETA {eta_str} | "
                        f"Elapsed {elapsed_str}"
                    )
                else:
                    msg = (
                        f"{total_processed}/{total_events} | "
                        f"{rate:,.0f} evt/s | "
                        f"Elapsed {elapsed_str}"
                    )
                print(msg, end="\r", file=sys.stdout)

                last_print = now
        print()# retour ligne propre
        print(f"Saved {file_out}")  
        logging.info(msg)
        logging.info(f"Final nreco / tot: {total_tracked}/{total_events}")
        output_dir = file_out.parent / "display"
        output_dir.mkdir(exist_ok=True, parents=True)
        if len(plots_dict) > 0:
            for evt_id, cfg_data in plots_dict.items():
                save_event_2d(evt_id, cfg_data, output_dir, z_panels=self.zpos, adc_ref=self.adc_ref)
            print(f"Saved {self.ndisplays} event displays to {output_dir}")  

    def _process_event(self, evt:Event, ndisplays=int):
        """
        Parameters
        ----------
        event : Event

        Returns
        -------
        rows : list of dict
            Une ligne par reconstruction valide (config)
        """
        rows = []
        plots = []
        # ----------------------------------
        # 0. GET XYZ (indices barreaux)
        # ----------------------------------
        xyz_ev = evt.xyz
        adc_ev = evt.adc
        if xyz_ev.shape[0] < 3:
            return rows, plots

        # hits_ev = np.column_stack((xyz_ev, (adc_ev[:,0]+adc_ev[:,1])/2))
        hits_ev =  np.column_stack((xyz_ev, adc_ev[:,0], adc_ev[:,1]))
        z_planes = np.unique(xyz_ev[:, 2])
        hits_by_plane = {
            int(z): hits_ev[hits_ev[:, 2] == z]
            for z in z_planes
        }
        # ----------------------------------
        # MULTIPLICITE PAR PANNEAU
        # ----------------------------------
        multiplicity = np.zeros(len(self.tel.panels), dtype=int)
        for z, hits in hits_by_plane.items():
            if z < len(multiplicity):
                multiplicity[z] = len(hits)

        for cfg_name, ixpos_cfg in self.dict_cfg_ix.items():
            # ----------------------------------
            # 1. SELECTION DES HITS
            # ----------------------------------
            # reconstruction du sous-ensemble xyz de la config
            hits_planes = [hits_by_plane[z] for z in ixpos_cfg if z in hits_by_plane]
            hits_cfg = np.vstack(hits_planes)
            if hits_cfg.shape[0] < 3:
                continue
            # vérifier nombre de plans distincts
            if len(np.unique(hits_cfg[:, 2])) < 3:
                continue
            # ----------------------------------
            # 2. NORMALISATION
            # ----------------------------------
            xyz_cfg = hits_cfg[:, :3]
            adc_cfg = hits_cfg[:, 3:]
            adc_cfg = np.asarray(adc_cfg)
            
            # xyz_norm, w_norm = normalize_xy_coords(xyz_cfg, 
            #                             pitch_xy=self.pitch_xy, 
            #                             pitch_ref=self.pitch_ref)
            xyz_cfg_mm = convert_bar_to_mm(
                xyz_cfg,
                self.arr_scint_width,   # pitch par plan
                self.arr_z_mm           # position Z en mm
            )

            # xyz_jit = apply_jitter(xyz_cfg_mm, scale=0.5) #jitter pour éviter les problèmes de colinéarité parfaite
            # w_norm = None
            # xyz_jit = apply_jitter(xyz_cfg, scale=0.5)
            z_cfg = xyz_cfg[:, 2].astype(int)
            pitch = self.arr_scint_width[z_cfg]
            sigma_jitter = pitch / np.sqrt(12)

            xyz_mm_jit = xyz_cfg_mm.copy()
            xyz_mm_jit[:, 0] += np.random.normal(0, sigma_jitter)
            xyz_mm_jit[:, 1] += np.random.normal(0, sigma_jitter)
            # if np.any(xyz_jit[:,0:2] <= 0.5): 
            #     print("evt.id:",evt.id)
            #     # print("xyz_norm_init:", xyz_norm_init)
            #     print("xyz_jit:",xyz_jit)
            # xyz_norm = xyz_cfg.copy()
            # hits_norm_jit = np.column_stack((xyz_norm_jit, adc_cfg)) #reconstruction de l'array hits avec les coordonnées normalisées et le adc
            use_ransac = True
            # ----------------------------------
            #  RANSAC 
            # ----------------------------------
            if use_ransac:
                ransac_result = self.ransac_model.fit(xyz_mm_jit)
                model = ransac_result

            if model is None:
                continue
                
            rms_ransac = model["rms"]
            inliers_idx = model["inliers_idx"]
            xyz_inliers = xyz_cfg[inliers_idx]
            xyz_inliers_mm = xyz_mm_jit[inliers_idx]
            # ----------------------------------
            # 5. INLIERS
            # ----------------------------------
            z_inliers = xyz_inliers[:,2]
            if len(np.unique(z_inliers)) < 3:
                continue
            # if len(xyz_inliers) < 3: continue
            # xyz_inliers = hits_cfg[inliers_idx]
            # ----------------------------------
            # 6. CONVERSION MM
            # ----------------------------------
            xyz_cfg_mm = convert_bar_to_mm(
                xyz_cfg,
                self.arr_scint_width,
                self.arr_z_mm, 
            )


            # ----------------------------------
            # 7. REFIT FINAL
            # ----------------------------------
            p0_ransac, v_ransac = fast_line_fit_2d(xyz_inliers_mm)
            p0_fin, v_fin = p0_ransac, v_ransac
            
            mip_score_x, mip_score_y = 0, 0
            if self.adc_ref is not None:           
                adc_inliers = adc_cfg[inliers_idx] 
                mpv_x = np.array([self.adc_ref[z]["x"]["mpv"] for z in z_inliers])
                mpv_y = np.array([self.adc_ref[z]["y"]["mpv"] for z in z_inliers])
                sigma_x = np.array([self.adc_ref[z]["x"]["sigma"] for z in z_inliers])
                sigma_y = np.array([self.adc_ref[z]["y"]["sigma"] for z in z_inliers])

                p0_weight, v_weight = weighted_line_fit_2d(xyz_inliers_mm, adc_inliers, mpv_ref=np.column_stack((mpv_x, mpv_y)), sigma_ref=np.column_stack((sigma_x, sigma_y)))
                
                # xyz_inliers_front = xyz_inliers[np.argmin(xyz_inliers[:, 2])]
                # xyz_inliers_rear = xyz_inliers[np.argmax(xyz_inliers[:, 2])]
                # rms_weight = np.sqrt(np.mean(np.linalg.norm(np.cross(np.vstack([xyz_inliers_front, xyz_inliers_rear]) - p0_weight, v_weight), axis=1)**2))
                rms_weight = compute_rms(xyz_inliers_mm, p0_weight, v_weight)

                p0_fin, v_fin = p0_weight, v_weight
                rms_fin = rms_weight

                # Approximation : même ADC projeté sur X et Y
                adc_x, adc_y = adc_inliers[:,0], adc_inliers[:,1]
                zscore_x = (adc_x - mpv_x) / (sigma_x + 1e-6)
                zscore_y = (adc_y - mpv_y) / (sigma_y + 1e-6)
                # zscore = 0.5 * (zscore_x + zscore_y)
                zscore_x, zscore_y = np.clip(zscore_x, -5, 5), np.clip(zscore_y, -5, 5)
                chi2_x, chi2_y = np.median(zscore_x**2), np.median(zscore_y**2)
                mip_score_x = np.exp(-0.5 * chi2_x)
                mip_score_y = np.exp(-0.5 * chi2_y)

            # if evt.id == 416:
            #     print(f"Event {evt.id} | Config {cfg_name} | MIP score X: {mip_score_x:.3f} | MIP score Y: {mip_score_y:.3f}")
            #     print(f"  Inliers Z: {z_inliers}")
            #     print(f"  xyz inliers (mm): {xyz_inliers_mm.tolist()}")
            
            if abs(v_fin[2]) < 1e-12:
                continue
            dz = v_fin[2] 
            dx_dz = v_fin[0] / dz
            dy_dz = v_fin[1] / dz
            # ----------------------------------
            # 7. OUTPUT
            # ----------------------------------
            pts = self.intersections(p0_weight, v_weight, cfg_name)  # shape (n_planes, 3)
            inside = int(
                np.all(
                    (pts[:, 0] >= 0) & (pts[:, 0] <= 800) &
                    (pts[:, 1] >= 0) & (pts[:, 1] <= 800)
                )
            ) #check if all intersection points are contained in telescope panels 
        
            if not inside: 
                continue
            
            rows.append((
                        evt.id,
                        int(evt.timestamp.s),
                        multiplicity[0],
                        multiplicity[1],
                        multiplicity[2],
                        multiplicity[3] if len(multiplicity) > 3 else 0,
                        cfg_name,
                        dx_dz,
                        dy_dz,
                        dz,
                        rms_fin,
                        int(use_ransac),
                        mip_score_x, mip_score_y
                        ))
            
            if ndisplays > 0:
                # ----------------------------------
                # BARYCENTRE PONDERE PAR ADC
                # ----------------------------------
                xyz_bary, weights_cfg_x, weights_cfg_y = barycenter_per_plane_2d(xyz_cfg, adc_cfg, self.adc_ref)
                
                xyz_bary_mm = convert_bar_to_mm(
                                xyz_bary,
                                self.arr_scint_width,
                                self.arr_z_mm
                            )
                plots.append ((
                    evt.id,
                    cfg_name,
                    (xyz_cfg_mm, xyz_bary_mm),
                    (weights_cfg_x, weights_cfg_y), 
                    inliers_idx,  
                    adc_cfg,
                    (p0_ransac, p0_weight),
                    (v_ransac, v_weight),
                    (rms_ransac, rms_weight),
                    int(use_ransac),
                    (mip_score_x, mip_score_y)
                ))
        return rows, plots

def compute_rms(xyz, p0, v):
    """
    Distance perpendiculaire point → droite
    """
    diff = xyz - p0
    cross = np.cross(diff, v)
    dist = np.linalg.norm(cross, axis=1)
    return np.sqrt(np.mean(dist**2))

def fast_line_fit(xyz):
    p0 = xyz.mean(axis=0)
    u, s, vh = np.linalg.svd(xyz - p0)
    v = vh[0]
    return p0, v

def fast_line_fit_2d(xyz):
    """
    Fit séparé XZ et YZ
    Retourne un p0 cohérent + direction reconstruite
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # --- FIT XZ : x = a*z + b ---
    A = np.vstack([z, np.ones_like(z)]).T
    ax, bx = np.linalg.lstsq(A, x, rcond=None)[0]

    # --- FIT YZ : y = c*z + d ---
    ay, by = np.linalg.lstsq(A, y, rcond=None)[0]

    # --- direction ---
    v = np.array([ax, ay, 1.0])
    v /= np.linalg.norm(v)

    # --- point d’origine (z moyen) ---
    z0 = np.mean(z)
    x0 = ax * z0 + bx
    y0 = ay * z0 + by
    p0 = np.array([x0, y0, z0])

    return p0, v

def compute_adaptive_threshold(xyz_mm, z_planes, arr_scint_width):
    # sigma par hit
    sigma_xy = arr_scint_width[z_planes] / np.sqrt(12)
    # fit rapide
    Z = xyz_mm[:, 2]
    A = np.vstack([Z, np.ones_like(Z)]).T
    try:
        ax, _ = np.linalg.lstsq(A, xyz_mm[:, 0], rcond=None)[0]
        ay, _ = np.linalg.lstsq(A, xyz_mm[:, 1], rcond=None)[0]
    except:
        return 30.0  # fallback
    # facteur angulaire
    theta_factor = np.sqrt(1 + ax**2 + ay**2)
    # sigma effectif
    sigma_eff = sigma_xy * theta_factor
    # robuste
    sigma_event = np.percentile(sigma_eff, 80)
    return 2.5 * sigma_event

def save_event_2d(event_id, config_data, output_dir, z_panels=None, adc_ref=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # couleurs fixes → lisibilité
    color_map = {
        "3p1": "blue",
        "3p2": "green",
        "4p": "red"
    }
    ax_xz, ax_yz = axes
    zmax = (len(z_panels)+1) * 600
    z_line = np.linspace(-100, zmax, 50)#[::-1]
    # Add MPV reference markers
    x_ref = -50  # position hors du plot pour ne pas gêner la visualisation des hits
    if adc_ref is not None:
        for i, z in enumerate(z_panels):
            if i in adc_ref:
                for coord, ax in {"x": ax_xz, "y": ax_yz}.items():
                    mpv = adc_ref[i][coord]["mpv"]
                    size = scale_adc_to_size(mpv)  # fixed size for marker
                    ax.scatter(x_ref, z, s=size, color='purple', alpha=0.7, edgecolor='black', label='MPV ref [ADC]' if i == 0 else "")
                    # add text with mpv value
                    ax.text(x_ref, z-50, f'{mpv:.0f}', fontsize=8, ha='center', va='bottom', color='black')
                    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

    for i, (cfg, ((xyz_cfg_mm, xyz_bary_mm), (weights_cfg_x, weights_cfg_y), inliers_idx, adc, (p0_ransac,p0_weight), (v_ransac,v_weight), (rms_ransac, rms_weight), use_ransac, (mip_score_x, mip_score_y))) in enumerate(config_data.items()):
        adc_x = adc[:,0]
        adc_y = adc[:,1]
        print(f"Plotting event {event_id} config {cfg} with {len(inliers_idx)} inliers (RMS: ({rms_ransac:.2f}, {rms_weight:.2f}), MIP: ({mip_score_x:.2f}, {mip_score_y:.2f})) ")
        # print(f"p0_ransac: {p0_ransac}, v_ransac: {v_ransac}")
        # print(f"p0_weight: {p0_weight}, v_weight: {v_weight}")
        mask_inliers = np.zeros(xyz_cfg_mm.shape[0], dtype=bool)
        mask_inliers[inliers_idx] = True
        color = color_map.get(cfg, None)
        # zmin, zmax = hits[:,2].min(), hits[:,2].max()
        sizes_xz= scale_adc_to_size(adc_x)
        sizes_yz= scale_adc_to_size(adc_y)
        # -------------------
        # XZ
        # -------------------
        if i == 0 : 
            ax_xz.scatter(
            xyz_cfg_mm[:,0], xyz_cfg_mm[:,2],
            alpha=0.6, color="darkgrey", edgecolor="black", linewidth=0.5, label="Hits with weights", 
            sizes=sizes_xz,
        )
            for j, (x, z) in enumerate(zip(xyz_cfg_mm[:,0], xyz_cfg_mm[:,2])):
                ax_xz.text(x, z-50, f"{weights_cfg_x[j]:.2f}", fontsize=6, ha="center", va="bottom", color="black", alpha=0.7)
            ax_xz.scatter(
            xyz_bary_mm[:,0], xyz_bary_mm[:,2],
            marker="X", s=100, alpha=0.8, color="orange", edgecolor="black", linewidth=1.5, label="Barycenter",  
        )
        ax_xz.scatter(
            xyz_cfg_mm[mask_inliers][:,0], xyz_cfg_mm[mask_inliers][:,2],
            s=sizes_xz[mask_inliers], alpha=0.6, color=color,
        )
        t_ransac = (z_line - p0_ransac[2]) / v_ransac[2]
        x_ransac = p0_ransac [0] + v_ransac[0]*t_ransac
        y_ransac = p0_ransac [1] + v_ransac[1]*t_ransac
        z_ransac = p0_ransac [2] + v_ransac[2]*t_ransac

        t_fin = (z_line - p0_weight[2]) / v_weight[2]
        x_fin = p0_weight [0] + v_weight[0]*t_fin
        y_fin = p0_weight [1] + v_weight[1]*t_fin
        z_fin = p0_weight [2] + v_weight[2]*t_fin

        ax_xz.plot(
            x_ransac, z_ransac,
            linewidth=2, label=f"{cfg} track ransac RMS_norm:{rms_ransac:.2f}", color=color,  linestyle="dashed"
        )
        ax_xz.plot(
            x_fin, z_fin, 
            linewidth=2, label=f"{cfg} track final RMS_mm:{rms_weight:.2f}, MIP:{mip_score_x:.2f}", color=color,
        )
        # -------------------
        # YZ
        # -------------------
        if i == 0 : 
            ax_yz.scatter(
                xyz_cfg_mm[:,1], xyz_cfg_mm[:,2],
                alpha=0.6, color="darkgrey", edgecolor="black", linewidth=0.5, label="Hits with weights", 
                sizes=sizes_yz,
            )
            for j, (y, z) in enumerate(zip(xyz_cfg_mm[:,1], xyz_cfg_mm[:,2])):
                ax_yz.text(y, z-50, f"{weights_cfg_y[j]:.2f}", fontsize=6, ha="center", va="bottom", color="black", alpha=0.7)
            ax_yz.scatter(
                xyz_bary_mm[:,1], xyz_bary_mm[:,2],
                marker="X", s=100, alpha=0.8, color="orange", edgecolor="black", linewidth=1.5, label="Barycenter",
                
            )
        ax_yz.scatter(
            xyz_cfg_mm[mask_inliers][:,1], xyz_cfg_mm[mask_inliers][:,2],
            s=sizes_yz[mask_inliers], alpha=0.6, color=color
        )

        ax_yz.plot(
            y_ransac, z_ransac,
            linewidth=2, label=f"{cfg} track ransac RMS_norm:{rms_ransac:.2f}", color=color, linestyle="dashed"
        )
        ax_yz.plot(
            y_fin, z_fin,
            linewidth=2, label=f"{cfg} track final RMS_mm:{rms_weight:.2f}, MIP:{mip_score_y:.2f}", color=color
        )


    # -------------------
    # STYLE
    # -------------------

    ax_xz.set_xlabel("X [mm]")
    ax_xz.set_ylabel("Z [mm]")
    ax_xz.set_title("XZ")
    ax_xz.set_xlim(-100, 900)
    
    ax_xz.set_ylim(-150, zmax)
    if z_panels : ax_xz.set_yticks(z_panels)
    ax_xz.invert_yaxis()
    ax_xz.legend(fontsize=8, loc="lower right")

    ax_yz.set_xlabel("Y [mm]")
    ax_yz.set_ylabel("Z [mm]")
    ax_yz.set_title("YZ")
    ax_yz.set_xlim(-100, 900)
    ax_yz.set_ylim(-150, zmax)
    if z_panels : ax_yz.set_yticks(z_panels)
    ax_yz.invert_yaxis()
    ax_yz.label_outer()
    ax_yz.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Event {event_id}")

    # ----------------------------------
    # SAVE
    # ----------------------------------
    filename = output_dir / f"event_{event_id:04d}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)  

def scale_adc_to_size(adc, smin=5, smax=250, adc_range=[0,3000]):
    adc = np.asarray(adc)
    if adc_range is None: 
        adc_range = (adc.min(), adc.max())
        if adc.max() == adc.min():
        # éviter division par 0
            return np.full_like(adc, (smin + smax)/2)
    adc_norm = (adc - adc_range[0]) / (adc_range[1] - adc_range[0])
    return smin + adc_norm * (smax - smin)

def barycenter_per_plane(xyz, adc, adc_ref, eps=1e-9):
    """
    Calcule la position barycentrique (x,y) pour chaque plan.

    Parameters
    ----------
    xyz : ndarray (N,3)
        Positions des hits (en mm ou indices convertis)
    adc : ndarray (N,)
        Valeurs ADC associées
    adc_ref : dict
        Références ADC par plan (ex: {0: {"mpv": 120, "sigma": 10}, 1: {"mpv": 95, "sigma": 8}, 2: {"mpv": 130, "sigma": 12}})
    eps : float
        Sécurité numérique

    Returns
    -------
    bary : ndarray (M,3)
        Positions barycentriques (x,y,z) pour chaque plan (M plans)
    weights : ndarray (N,)
        poids calculés pour chaque hit
    """

    xyz = np.asarray(xyz)
    adc = np.asarray(adc)
    z_planes = xyz[:,2]
    z_unique = np.unique(z_planes)
    # Poids gaussiens centrés sur la MPV du plan
    adc_mpv_ref = np.array([ adc_ref.get(int(z), 1.0)["mpv"] for z in z_planes ])
    # adc_norm = adc / (adc_mpv_ref + 1e-6)
    # adc_med = np.median(adc_norm)
    adc_sigma_ref = np.array([ adc_ref.get(int(z), 1.0)["sigma"] for z in z_planes ])  # tolérance MIP
    # weights = np.exp(-0.5 * ((adc_norm - adc_med)/adc_sigma_ref)**2)
    weights = 1.0 / (1.0 + ((adc - adc_mpv_ref) / adc_sigma_ref) ** 2)
    # weights /= (weights.sum() + 1e-12)
    # Calcul barycentres
    bary = np.zeros((len(z_unique), 3))  # (n_planes, 2)
    for i, z in enumerate(z_unique):
        mask = z_planes == z
        w = weights[mask]
        w_sum = np.sum(w) + eps
        x_bary = np.sum(xyz[mask, 0] * w) / w_sum
        y_bary = np.sum(xyz[mask, 1] * w) / w_sum
        bary[i] = (x_bary, y_bary, z)
    return bary, weights

def barycenter_per_plane_2d(xyz, adc, adc_ref, eps=1e-9):
    """
    Barycentre par plan avec poids séparés X/Y

    Returns
    -------
    bary : ndarray (M,3)
        (x_bary, y_bary, z)
    weights_x : ndarray (N,)
    weights_y : ndarray (N,)
    """

    xyz = np.asarray(xyz)
    adc = np.asarray(adc)
    adc_x, adc_y = adc[:,0], adc[:,1]

    z_planes = xyz[:, 2].astype(int)
    z_unique = np.unique(z_planes)

    # --- MPV / sigma séparés X et Y ---
    mpv_x = np.array([adc_ref[z]["x"]["mpv"] for z in z_planes])
    sigma_x = np.array([adc_ref[z]["x"]["sigma"] for z in z_planes])

    mpv_y = np.array([adc_ref[z]["y"]["mpv"] for z in z_planes])
    sigma_y = np.array([adc_ref[z]["y"]["sigma"] for z in z_planes])

    # --- poids robustes (type Cauchy) ---
    weights_x = 1.0 / (1.0 + ((adc_x - mpv_x) / (sigma_x + eps))**2)
    weights_y = 1.0 / (1.0 + ((adc_y - mpv_y) / (sigma_y + eps))**2)

    # --- barycentres ---
    bary = np.zeros((len(z_unique), 3))

    for i, z in enumerate(z_unique):
        mask = z_planes == z

        # X pondéré par calibration X
        wx = weights_x[mask]
        wx_sum = np.sum(wx) + eps
        x_bary = np.sum(xyz[mask, 0] * wx) / wx_sum

        # Y pondéré par calibration Y
        wy = weights_y[mask]
        wy_sum = np.sum(wy) + eps
        y_bary = np.sum(xyz[mask, 1] * wy) / wy_sum

        bary[i] = (x_bary, y_bary, z)

    return bary, weights_x, weights_y

def weighted_fit(xyz, weights):
    w = weights / weights.sum()
    centroid = np.sum(w[:,None] * xyz, axis=0)
    X = xyz - centroid
    cov = (w[:,None,None] * (X[:,:,None] @ X[:,None,:])).sum(axis=0)
    _, _, vh = np.linalg.svd(cov)
    direction = vh[0]
    return centroid, direction

def weighted_fit_mip(xyz_inliers_mm, z_inliers, adc_inliers, adc_ref:dict):
    adc_mpv_ref = np.array([ adc_ref.get(int(z), 1.0)["mpv"] for z in z_inliers ])
    # adc_norm = adc_inliers / (adc_mpv_ref + 1e-6)
    adc_med = np.median(adc_mpv_ref)
    sigma = np.array([ adc_ref.get(int(z), 1.0)["sigma"] for z in z_inliers ])  # tolérance MIP
    weights = np.exp(-0.5 * ((adc_mpv_ref - adc_med)/sigma)**2)
    weights /= (weights.sum() + 1e-12)
    # weighted PCA
    centroid = np.sum(weights[:,None] * xyz_inliers_mm, axis=0)
    X = xyz_inliers_mm - centroid
    cov = (weights[:,None,None] * (X[:,:,None] @ X[:,None,:])).sum(axis=0)
    _, _, vh = np.linalg.svd(cov)
    v = vh[0]
    p0 = centroid
    return p0, v, weights


def weighted_line_fit_2d(xyz, adc, mpv_ref, sigma_ref, eps=1e-9):
    """
    Fit pondéré séparé XZ / YZ

    Parameters
    ----------
    xyz : ndarray (N,3)
    z_panels : ndarray (N,)
    adc : ndarray (N,)
    adc_ref : dict

    Returns
    -------
    p0 : ndarray (3,)
    v  : ndarray (3,)
    """

    xyz = np.asarray(xyz)
    adc = np.asarray(adc)
    adc_x, adc_y = adc[:,0], adc[:,1]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2].astype(int)

    # --- récupérer MPV / sigma ---
    # mpv_x = np.array([adc_ref[zi]["x"]["mpv"] for zi in z_panels])
    # sigma_x = np.array([adc_ref[zi]["x"]["sigma"] for zi in z_panels])

    # mpv_y = np.array([adc_ref[zi]["y"]["mpv"] for zi in z_panels])
    # sigma_y = np.array([adc_ref[zi]["y"]["sigma"] for zi in z_panels])
    mpv_x, mpv_y = mpv_ref[:,0], mpv_ref[:,1]
    sigma_x, sigma_y = sigma_ref[:,0], sigma_ref[:,1]
    # --- poids robustes ---
    wx = 1.0 / (1.0 + ((adc_x - mpv_x) / (sigma_x + eps))**2)
    wy = 1.0 / (1.0 + ((adc_y - mpv_y) / (sigma_y + eps))**2)

    # --- matrice design ---
    A = np.vstack([z, np.ones_like(z)]).T  # (N,2)

    # --- XZ pondéré ---
    Awx = A * wx[:, None]
    ax, bx = np.linalg.lstsq(Awx, x * wx, rcond=None)[0]

    # --- YZ pondéré ---
    Awy = A * wy[:, None]
    ay, by = np.linalg.lstsq(Awy, y * wy, rcond=None)[0]

    # --- direction ---
    v = np.array([ax, ay, 1.0])
    v /= np.linalg.norm(v)

    # --- point origine ---
    z0 = np.mean(z)
    x0 = ax * z0 + bx
    y0 = ay * z0 + by
    p0 = np.array([x0, y0, z0])

    return p0, v


def merge_adc_dicts(list_of_dicts, n_planes):
    merged = {z: [] for z in range(n_planes)}
    for d in list_of_dicts:
        for z in range(n_planes):
            merged[z].extend(d[z])
    return merged

def _tracking_chunk_static(chunk, config, dicts, models_config):
    tracker = config["tracker_class"](**config["init"])
    tracker.dict_id_ixpos = dicts["ixpos"]
    # tracker.dict_id_zpos  = dicts["zpos"]
    # ----------------------------------
    # MODELS INITIALIZED ONCE
    # ----------------------------------
    tracker.det_model = DeterministicModel(
        nx=models_config["det"]["nx"],
        ny=models_config["det"]["ny"],
        residual_threshold=models_config["det"]["residual_threshold"]
    )
    ransac_kwargs = models_config["ransac"]
    tracker.ransac_model = RansacModel(**ransac_kwargs)
    rows = []
    plots = []
    ntot, nprocess, ntrack = 0,0, 0
    for evt in chunk:
        evt.get_xyz(dict_ixpos=tracker.dict_id_ixpos)
        # print(f"Processing event {evt.id} with {len(evt.xyz)} hits")
        if len(evt.xyz) == 0:
            ntot +=1
            continue
        if not tracker.prefilter(evt):
            ntot +=1
            continue
        rows_evt, plots_evt = tracker._process_event(evt, config["ndisplays"])
        if len(rows_evt) >0 :
            rows.extend(rows_evt)
            ntrack += 1
        if len(plots_evt) > 0:
            plots.extend(plots_evt)
        ntot +=1
        nprocess += 1
    return rows, plots, ntot, nprocess, ntrack