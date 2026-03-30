#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import time

#package modules
from cli import get_processing_args
from data import RawData
from tracking import RansacTracking
from utils.common import Common

t0 = time.strftime("%H:%M:%S", time.localtime())
args = get_processing_args()
cmn = Common(args)
survey = cmn.survey
tel = cmn.telescope
run = cmn.run 
start_time = time.time()
strdate = time.strftime("%d%m%Y_%H%M")
flog =str(run.dirs["log"]/f'{strdate}.log')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    filemode='w',
                    filename=flog,)
logging.info(vars(args))
logging.info(f"Start -- {t0}")
kwargs_ransac = dict(residual_threshold=args.residual_threshold, 
            min_samples=args.min_samples, 
            max_trials=args.max_trials,  
) 
in_dir = run.dirs["raw"]
raw = RawData(path=in_dir)
raw.fill_dataset(nfiles = int(args.nfiles))
print(f"file0 : {raw.dataset[0]}")
print(f"Read nfiles / tot : {len(raw.dataset)} / {raw.nfiles_tot}")
tracking = RansacTracking(telescope = tel, data = raw, entry_start=0, nev_max=args.nevents )
ftrack = run.dirs["reco"] / 'df_track.csv.gz'
tracking.main(file_out=ftrack, **kwargs_ransac)
logging.info(f"Saved {ftrack}")
t_sec = round(time.time() - start_time)
(t_min, t_sec) = divmod(t_sec,60)
(t_hour,t_min) = divmod(t_min,60)
t_end = 'Total Elapsed : {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
logging.info(t_end)
print(f"Saved log {flog}")