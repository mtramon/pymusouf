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
start_time = time.time()
strdate = time.strftime("%d%m%Y_%H%M")
flog =str(cmn.log_path/f'{strdate}.log')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    filemode='w',
                    filename=flog,)
                    # stream=sys.stdout,) #either set 'filename' to save info in log file or 'stream' to print out on console
logging.info(vars(args))
logging.info(f"Start -- {t0}")
kwargs_ransac = dict(residual_threshold=args.residual_threshold, 
            min_samples=args.min_samples, 
            max_trials=args.max_trials,  
) 
logging.info('Ransac Tracking')
in_dir = cmn.raw_path
raw = RawData(path=in_dir)
raw.fill_dataset(nfiles = int(args.nfiles))
print(f"file_0 : {raw.dataset[0]}")
print(f"nfiles_dataset = {len(raw.dataset)}")
# file="OrMec_FDN_TOMO4_20170717-7"
# raw.dataset = [ f for f in raw.dataset if file in str(f)]
# print(raw.dataset)
tracking = RansacTracking(telescope = tel, data = raw, entry_start=0, nev_max=args.nevents )
out_dir = cmn.reco_path
ftrack = out_dir / 'df_track.csv.gz'
tracking.main(file_out=ftrack, **kwargs_ransac)
logging.info(tracking)
tracking.df_track.to_csv(ftrack, compression='gzip', index=False)
logging.info(f"Save dataframe {ftrack}")
# print(f"df_track.head = {tracking.df_track.head}")

# fmodel = out_dir / 'df_inlier.csv.gz' #ransac inlier pt-tagging output for all reco events
# tracking.df_model.to_csv(fmodel, compression='gzip', index=False, sep='\t')
# logging.info(f"Save dataframe {fmodel}")
t_sec = round(time.time() - start_time)
(t_min, t_sec) = divmod(t_sec,60)
(t_hour,t_min) = divmod(t_min,60)
t_end = 'Duration : {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
logging.info(t_end)

logging.info(f"Output directory : {out_dir}")
print(f"Log: {flog}")