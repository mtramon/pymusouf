#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import argparse
import time
import logging
import pandas as pd
from multiprocessing import Pool
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import cProfile
import pstats
import io

#personal modules
from survey import CURRENT_SURVEY
from survey.data import RawData
from survey.run import Run
from telescope import DICT_TEL,  str2telescope
from tracking import RansacModel, RansacTracking
from utils.tools import str2bool, print_progress

def process_file(praw, args, kwargs_ransac, out_dir, n):
    try:
        raw = RawData(path=praw)
        run = Run(name=praw, telescope=args.telescope, rawdata=[raw])
        raw.fill_dataset(max_nfiles=args.max_nfiles)
        tracking = RansacTracking(telescope=args.telescope, data=raw)
        tracking.process(model_type=RansacModel, progress_bar=args.progress_bar, **kwargs_ransac)
        
        ftrack = str(out_dir) + '/' + 'df_track' + '_' + str(n) + '.csv.gz'
        tracking.df_track.to_csv(ftrack, compression='gzip', index=False, sep='\t')

        fmodel = str(out_dir) + '/' + 'df_inlier' + '_' + str(n) + '.csv.gz'
        tracking.df_model.to_csv(fmodel, compression='gzip', index=False, sep='\t')

        return ftrack, fmodel
    except Exception as e:
        logging.error(f"Failed to process file {praw}: {e}")
        logging.error(traceback.format_exc())
        return None, None

def process_wrapper(args):
    return process_file(*args)

if __name__ == "__main__":

    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    t0 = time.strftime("%H:%M:%S", time.localtime())
    print("Start: ", t0)#start time
    t_start = time.perf_counter()
    home_path = Path.home()
    parser=argparse.ArgumentParser(
    description='''For a given muon telescope configuration, this script allows to perform RANSAC tracking and outputs trajectrory-panel crossing XY coordinates''', epilog="""All is well that ends well.""")
    parser.add_argument('--telescope', '-tel', required=True, help='Input telescope name. It provides the associated configuration.', type=str2telescope)
    parser.add_argument('--input_data', '-i', nargs="*", required=True, help='/path/to/datafile/  One can input a data directory, a single datfile, or a list of data files e.g "--input_data <file1.dat> <file2.dat>"', type=str)
    parser.add_argument('--out_dir', '-o', required=True, help='Path to processing output', type=str) 
    parser.add_argument('--input_type', '-it', default='real',  help="'real' or 'mc'", type=str)
    parser.add_argument('--max_nfiles', '-max', default=1, help='Maximum number of dataset files to process.', type=int)
    parser.add_argument('--residual_threshold', '-rt', default=50, help="RANSAC 'distance-to-model' parameter in mm",type=float)
    parser.add_argument('--min_samples', '-ms', default=2, help='RANSAC size of the initial sample',type=int)
    parser.add_argument('--max_trials', '-mt', default=100, help='RANSAC number of iterations',type=int)
    parser.add_argument('--fit_intersect', '-intersect', default=False, help='if true record line model intersection points on panel; else record closest XY inlier points to model',type=str2bool)
    parser.add_argument('--info', '-info', default=None, help='Additional info',type=str)
    parser.add_argument('--progress_bar', '-bar', default=False, help='Display progress bar',type=str2bool)
    parser.add_argument('--num_workers', '-nw', default=os.cpu_count(), help='Number of workers for parallel processing', type=int)
    args=parser.parse_args()

    # survey = CURRENT_SURVEY[args.telescope.name]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    reco_dir = out_dir 
    reco_dir.mkdir(parents=True, exist_ok=True)

    strdate = time.strftime("%d%m%Y_%H%M")
    flog =str(out_dir/f'{strdate}.log')
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        filemode='w',
                        #filename=flog,)
                        stream=sys.stdout,) #either set 'filename' to save info in log file or 'stream' to print out on console
    logging.info(sys.argv)
    logging.info(f"Start -- {t0}")
    if args.info : logging.info(args.info) #additional info

    kwargs_ransac = dict(residual_threshold=args.residual_threshold, 
                min_samples=args.min_samples, 
                max_trials=args.max_trials,  
    ) 

    # Fix rawdata_path to detect all files in the directory
    rawdata_path = []
    for p in args.input_data:
        p = Path(p)
        if p.is_dir():
            rawdata_path.extend(p.glob('*.dat.gz'))
            rawdata_path.extend(p.glob('*.npy'))
        else:
            rawdata_path.append(p)

    logging.info(f"Number of files to be read: {len(rawdata_path)}")
    total_files = len(rawdata_path)
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_file, praw, args, kwargs_ransac, out_dir, n) for n, praw in enumerate(rawdata_path)]
        for i, future in enumerate(as_completed(futures)):
            ftrack, fmodel = future.result()
            if ftrack and fmodel:
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (i + 1)) * (total_files - (i + 1))
                logging.info(f"Progress: {((i + 1) / total_files) * 100:.2f}% - Estimated time remaining: {remaining_time / 60:.2f} minutes")

    # Merge all df_inlier*.csv.gz files into one df_inlier.csv.gz
    df_inlier_files = list(out_dir.glob('df_inlier_*.csv.gz'))
    if df_inlier_files:
        df_inlier_list = [pd.read_csv(f, compression='gzip', sep='\t') for f in df_inlier_files]
        df_inlier = pd.concat(df_inlier_list, ignore_index=True)
        df_inlier.to_csv(out_dir / 'df_inlier.csv.gz', compression='gzip', index=False, sep='\t')
        logging.info(f"Merged {len(df_inlier_files)} df_inlier files into df_inlier.csv.gz")

    # Merge all df_track*.csv.gz files into one df_track.csv.gz
    df_track_files = list(out_dir.glob('df_track_*.csv.gz'))
    if df_track_files:
        df_track_list = [pd.read_csv(f, compression='gzip', sep='\t') for f in df_track_files]
        df_track = pd.concat(df_track_list, ignore_index=True)
        df_track.to_csv(out_dir / 'df_track.csv.gz', compression='gzip', index=False, sep='\t')
        logging.info(f"Merged {len(df_track_files)} df_track files into df_track.csv.gz")

    # Stop profiling
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open(str(out_dir / 'profiling_stats.txt'), 'w') as f:
        f.write(s.getvalue())

    t_sec = round(time.time() - start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    t_end = 'Duration : {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
    print(t_end)

    logging.info(t_end)

    logging.info(f"Output directory : {reco_dir}")
