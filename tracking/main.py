#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import argparse
import time
import logging
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import cProfile
import pstats
import io
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

# personal modules
from survey import CURRENT_SURVEY
from survey.data import RawData
from survey.run import Run
from telescope import str2telescope
from tracking import RansacModel, RansacTracking
from utils.tools import str2bool

# Configure logging
log_queue = Queue()
queue_handler = QueueHandler(log_queue)
handler = logging.StreamHandler(sys.stdout)
file_handler = None
listener = None

def configure_logging(log_file):
    global file_handler, stream_handler, listener

    # Define the desired format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a logger with handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set the global log level to DEBUG
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_format)  # Set format for file handler
    file_handler.setLevel(logging.DEBUG)  # Set file handler level to DEBUG

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)  # Set format for stream handler
    stream_handler.setLevel(logging.DEBUG)  # Set stream handler level to DEBUG

    # Add handlers to QueueListener
    listener = QueueListener(log_queue, file_handler, stream_handler)
    listener.start()

    # Use QueueHandler for the root logger
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    # Inform about the successful logging setup
    root_logger.info("Logging is configured.")

def process_file(praw, args, kwargs_ransac, out_dir, n):
    # logging.info(f"Processing file {praw}")
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    try:
        raw = RawData(path=praw)
        Run(name=praw, telescope=args.telescope, rawdata=[raw])
        raw.fill_dataset(max_nfiles=args.max_nfiles)
        tracking = RansacTracking(telescope=args.telescope, data=raw)
        tracking.process(model_type=RansacModel, progress_bar=args.progress_bar, **kwargs_ransac)
        
        ftrack = str(out_dir / f'df_track_{n}.csv.gz')
        if tracking.df_track.empty:
            logging.warning(f"df_track is empty for file {praw}")
        tracking.df_track.to_csv(ftrack, compression='gzip', index=False, sep='\t')

        fmodel = str(out_dir / f'df_inlier_{n}.csv.gz')
        if tracking.df_model.empty:
            logging.warning(f"df_inlier is empty for file {praw}")
        tracking.df_model.to_csv(fmodel, compression='gzip', index=False, sep='\t')

        return ftrack, fmodel
    except ValueError as e:
        logging.error(f"Aborting due to error: {e}")
        raise  # Re-raise the exception to ensure it is caught in the main execution flow
    except Exception as e:
        logging.error(f"Failed to process file {praw}: {e}")
        logging.error(traceback.format_exc())
        return None, None
    finally:
        if args.profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            with open(out_dir / f'profiling_stats_{n}.txt', 'w') as f:
                f.write(s.getvalue())

def process_wrapper(args):
    try:
        return process_file(*args)
    except ValueError as e:
        logging.error(f"Critical error: {e}. Aborting.")
        sys.exit(1)  # Immediate exit for critical errors
    except Exception as e:
        logging.error(f"Unexpected error: {e}.")
        return None, None

def merge_csv_files(file_pattern, output_file, out_dir):
    files = list(out_dir.glob(file_pattern))
    if files:
        df_list = []
        for f in files:
            try:
                df = pd.read_csv(f, compression='gzip', sep='\t')
                if not df.empty:
                    df_list.append(df)
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file skipped: {f}")
        if df_list:
            df_merged = pd.concat(df_list, ignore_index=True)
            df_merged.to_csv(out_dir / output_file, compression='gzip', index=False, sep='\t')
            logging.info(f"Merged {len(df_list)} files into {output_file}")
            for f in files:
                f.unlink()  # Delete the original file

if __name__ == "__main__":
    start_time = time.time()
    t0 = time.strftime("%H:%M:%S", time.localtime())
    
    parser = argparse.ArgumentParser(
        description='''For a given muon telescope configuration, this script allows to perform RANSAC tracking and outputs trajectory-panel crossing XY coordinates''',
        epilog="""All is well that ends well."""
    )
    parser.add_argument('--telescope', '-tel', required=True, help='Input telescope name. It provides the associated configuration.', type=str2telescope)
    parser.add_argument('--input_data', '-i', nargs="*", required=True, help='/path/to/datafile/  One can input a data directory, a single datafile, or a list of data files e.g "--input_data <file1.dat> <file2.dat>"', type=str)
    parser.add_argument('--out_dir', '-o', required=True, help='Path to processing output', type=str)
    parser.add_argument('--input_type', '-it', default='real', help="'real' or 'mc'", type=str)
    parser.add_argument('--max_nfiles', '-max', default=1, help='Maximum number of dataset files to process.', type=int)
    parser.add_argument('--residual_threshold', '-rt', default=50, help="RANSAC 'distance-to-model' parameter in mm", type=float)
    parser.add_argument('--min_samples', '-ms', default=2, help='RANSAC size of the initial sample', type=int)
    parser.add_argument('--max_trials', '-mt', default=100, help='RANSAC number of iterations', type=int)
    parser.add_argument('--fit_intersect', '-intersect', default=False, help='if true record line model intersection points on panel; else record closest XY inlier points to model', type=str2bool)
    parser.add_argument('--info', '-info', default=False, help='Additional info', type=str)
    parser.add_argument('--progress_bar', '-bar', default=False, help='Display progress bar', type=str2bool)
    parser.add_argument('--num_workers', '-nw', default=os.cpu_count(), help='Number of workers for parallel processing', type=int)
    parser.add_argument('--profile', '-p', default=False, help='Enable profiling', type=str2bool)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strdate = time.strftime("%d%m%Y_%H%M")
    flog = out_dir / f'{strdate}.log'

    configure_logging(flog)

    logging.info("Starting the main process.")
    logging.info(sys.argv)
    logging.info(f"Start -- {t0}")
    if args.info:
        logging.info(args.info)  # additional info

    # Start profiling
    pr = None
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    kwargs_ransac = dict(
        residual_threshold=args.residual_threshold,
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

    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_wrapper, (praw, args, kwargs_ransac, out_dir, n))
                for n, praw in enumerate(rawdata_path)
            ]
            
            for n, future in enumerate(as_completed(futures)):
                try:
                    ftrack, fmodel = future.result()
                    if (n + 1) % max(1, total_files // 100) == 0:
                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / (n + 1)) * (total_files - (n + 1))
                        logging.info(f"Progress: {((n + 1) / total_files) * 100:.2f}% - Estimated time remaining: {remaining_time / 60:.2f} minutes")
                except ValueError as e:
                    logging.error(f"Execution aborted due to error: {e}")
                    listener.stop()  # Stop logging
                    sys.exit(1)  # Immediate exit on error
                except Exception as e:
                    logging.error(f"Unexpected critical error: {e}")
                    listener.stop()  # Stop logging
                    sys.exit(1)  # Immediate exit on error
    except ValueError as e:
        logging.error(f"Execution aborted: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected critical error: {e}")
        sys.exit(1)

    # Merge all df_inlier*.csv.gz files into one df_inlier.csv.gz and delete original files
    merge_csv_files('df_inlier_*.csv.gz', 'df_inlier.csv.gz', out_dir)

    # Merge all df_track*.csv.gz files into one df_track.csv.gz and delete original files
    merge_csv_files('df_track_*.csv.gz', 'df_track.csv.gz', out_dir)

    # Stop profiling
    if args.profile:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open(out_dir / 'profiling_stats.txt', 'w') as f:
            f.write(s.getvalue())

    t_sec = round(time.time() - start_time)
    t_min, t_sec = divmod(t_sec, 60)
    t_hour, t_min = divmod(t_min, 60)
    t_end = f'Duration : {t_hour}hour:{t_min}min:{t_sec}sec'
    print(t_end)

    logging.info(t_end)
    logging.info(f"Output directory : {out_dir}")

    # Stop the QueueListener to ensure logs are flushed to the file
    listener.stop()
