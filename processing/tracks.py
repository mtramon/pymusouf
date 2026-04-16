#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import logging
import time
#package module(s)
from calibration import run_adc_calibration
from cli import get_processing_args
from data import RawData
from tracking import EventStream, RansacTracking
from utils.common import Common


DEFAULT_CHUNK_SIZE = 8000 

if __name__ == "__main__":
    t0 = time.strftime("%H:%M:%S", time.localtime())
    args = get_processing_args()
    cmn = Common(args)
    survey = cmn.survey
    tel = cmn.telescope
    run = cmn.run
    start_time = time.time()
    strdate = time.strftime("%d%m%Y_%H%M")
    flog = str(run.dirs["log"] / f"{strdate}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
        filename=flog,
    )
    logging.info(vars(args))
    logging.info(f"Start -- {t0}")

    kwargs_ransac = dict(
        residual_threshold=args.residual_threshold,
        min_samples=args.min_samples,
        max_trials=args.max_trials,
    )

    in_dir = run.dirs["raw"]
    print(in_dir)
    raw = RawData(path=in_dir)
    raw.fill_dataset(nfiles=int(args.nfiles))
    print(f"file0 : {raw.dataset[0]}")
    print(f"Read nfiles / tot : {len(raw.dataset)} / {raw.nfiles_tot}")
    adc_ref = None
    if args.adc_calibration:
        '''
        Calibration on MIP-like events to get the ADC reference values for each plane in X and Y. This is used to convert the ADC values to MIP units in the tracking step and associate weights to the hits.
        '''
        event_stream = EventStream(telescope=tel, data=raw, entry_start=0, nev_max=args.nevents)
        n_planes = len(tel.panels)
        adc_ref = run_adc_calibration(
            tel=tel,
            run=run,
            event_stream=event_stream,
            n_planes=n_planes,
            chunk_size=DEFAULT_CHUNK_SIZE,
            bins=50,
        )

    tracker = RansacTracking(
        telescope=tel,
        adc_ref=adc_ref,
        ndisplays=args.ndisplays,
    )
    event_stream = EventStream(telescope=tel, data=raw, entry_start=0, nev_max=args.nevents)
    chunks_tracking = event_stream.chunked(DEFAULT_CHUNK_SIZE)
    fout_csv = run.dirs["reco"] / "df_track.csv.gz"
    tracker.main(file_out=fout_csv, chunks=chunks_tracking, **kwargs_ransac)
    logging.info(f"Saved {fout_csv}")
    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    t_end = "Total Elapsed : {}hour:{}min:{}sec".format(t_hour, t_min, t_sec)
    logging.info(t_end)
    print(f"Saved log {flog}")
