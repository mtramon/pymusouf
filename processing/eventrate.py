#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from math import ceil
import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage import uniform_filter1d  # pour le lissage

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':5,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid' : True,
         }
plt.rcParams.update(params)



class EventRate:

    def __init__(self, time:np.ndarray, dt_gap:int=3600, t0=0):
        self.time= time
        self.run_duration = 0    
        self.time = time #s
        self.t0 = t0 

    def __call__(self, ax, width:float=3600, label:str="",  t_off:float=0., tlim=None, **kwargs):
        time_off = (self.time - min(self.time)) + self.t0 
        range_time_off = [np.nanmin(time_off),np.nanmax(time_off)]
        t_start, t_end = range_time_off
        self.run_duration = t_end - t_start
        ntbins = ceil((t_end-t_start) / width)
        entries, bins = np.histogram(time_off[~np.isnan(time_off)],   bins=ntbins)
        bins_c = (bins[1:]+bins[:-1])/2
        bins_w = (bins[1:]-bins[:-1])
        if t_start == t_end : t_end += self.run_duration
        date_start = datetime.fromtimestamp(t_start+t_off)
        date_end = datetime.fromtimestamp(t_end+t_off)
        self.date_start, self.date_end=  date_start, date_end
        n_events = len(self.time)
        label += (f"\nstart: {self.date_start.strftime('%y/%m/%d %H:%M')}"
                  f"\nend: {self.date_end.strftime('%y/%m/%d %H:%M')}"
                  f"\nduration = {self.run_duration // 3600:.0f}h{self.run_duration % 3600 / 60:.0f}min"
                  f"\nnevts = {n_events:.3e}"
                  f"\nrate = {n_events / self.run_duration:.2f} Hz")
        rate = entries/width #Hz
        self.bars = ax.bar(bins_c, rate, bins_w, label=label, **kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel("Event rate [Hz]")
        self.entries, self.bins = entries, bins_c
    
    def time_series(self, ax, width: float = 3600., window: int = 5, label: str = "", t_off: float = 0., tlim=None, **kwargs):
        time_off = (self.time - self.time[0]) + self.t0
        range_time_off = [np.nanmin(time_off), np.nanmax(time_off)]
        t_start, t_end = range_time_off
        self.run_duration = t_end - t_start
        ntbins = ceil((t_end - t_start) / width)

        # Histogramme avec bins réguliers
        entries, bins = np.histogram(time_off[~np.isnan(time_off)], bins=ntbins)
        bins_c = (bins[1:] + bins[:-1]) / 2
        bins_w = bins[1:] - bins[:-1]
        # Taux d'événements
        rate = entries / bins_w  # en Hz (événements/s)
        # Lissage avec fenêtre glissante (taille = window bins)
        rate_smooth = uniform_filter1d(rate, size=window)
        # Conversion timestamp pour affichage
        date_start = datetime.fromtimestamp(t_start + t_off)
        date_end = datetime.fromtimestamp(t_end + t_off)
        self.date_start, self.date_end = date_start, date_end
        # Tracé courbe lissée
        bins_c_days = bins_c / 86400  # seconds → days
        self.line, = ax.plot(bins_c_days, rate_smooth, label=label, **kwargs)
        ax.fill_between(bins_c_days, rate_smooth, alpha=0.3, color=self.line.get_color())
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Event rate [Hz]")
        # self.entries, self.bins = entries, bins_c
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((-2, 2))
        # ax.xaxis.set_major_formatter(formatter)
        return bins_c_days, rate_smooth

if __name__ == "__main__":


    pass
