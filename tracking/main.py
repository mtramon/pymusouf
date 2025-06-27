#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import argparse
import time
import logging
import logging.config
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import cProfile
import pstats
import io
from multiprocessing import Queue

# personal modules
from survey import CURRENT_SURVEY
from survey.data import RawData
from survey.run import Run
from telescope import str2telescope
from tracking import RansacModel, RansacTracking
from utils.tools import str2bool

# Configure logging
# log_queue = Queue()
# queue_handler = QueueHandler(log_queue)
# handler = logging.StreamHandler(sys.stdout)
# file_handler = None
# listener = None

def configure_logging(log_file: str,
                      log_level: str = "INFO",
                      when: str = "midnight",
                      backup_count: int = 7):
    """
    Configura logging y devuelve (queue, listener) para poder
    compartir la cola con procesos hijos.
    """
    from multiprocessing import get_context
    ctx = get_context("spawn")
    log_queue = ctx.Queue()

    # Handler que rueda el archivo cada día a medianoche y guarda 7 backups
    rotating_handler = TimedRotatingFileHandler(
        log_file, when=when, backupCount=backup_count, encoding="utf-8"
    )
    fmt = "%(asctime)s | %(levelname)-8s | %(processName)s | %(funcName)s:%(lineno)d | %(message)s"
    rotating_handler.setFormatter(logging.Formatter(fmt))
    rotating_handler.setLevel(logging.DEBUG)  # Archivo: todo (DEBUG+)

    # Console handler: solo WARNING y ERROR a la terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    console_handler.setLevel(logging.WARNING)  # Terminal: solo WARNING+

    # Ambos handlers al listener
    listener = QueueListener(log_queue, rotating_handler, console_handler)
    listener.start()

    # Diccionario base – el root logger envía todo a la cola
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "console_warning_filter": {
                "()": "logging.Filter",
                "name": "",
            }
        },
        "handlers": {
            "queue": {
                "class": "logging.handlers.QueueHandler",
                "queue": log_queue,
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["queue"],
        },
        "loggers": {
            "tracking": {
                "level": "INFO",
                "propagate": True,
                "handlers": []
            }
        }
    }
    logging.config.dictConfig(config)

    # Refuerza el nivel del handler de consola después de la configuración
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.WARNING)

    return log_queue, listener

# Configuración del logger (llamado al iniciar el script principal)
def init_logger(log_level="INFO", log_retention=7):
    strdate = time.strftime("%d%m%Y_%H%M")
    log_dir = Path("output_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    flog = log_dir / f'{strdate}.log'
    log_queue, listener = configure_logging(flog, log_level=log_level, backup_count=log_retention)
    return listener

def process_file(praw, args, kwargs_ransac, out_dir, n):
    logger = logging.getLogger(__name__)
    # logger.info(f"Processing file {praw}")
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
            logger.warning(f"df_track is empty for file {praw}")
        tracking.df_track.to_csv(ftrack, compression='gzip', index=False, sep='\t')

        fmodel = str(out_dir / f'df_inlier_{n}.csv.gz')
        if tracking.df_model.empty:
            logger.warning(f"df_inlier is empty for file {praw}")
        tracking.df_model.to_csv(fmodel, compression='gzip', index=False, sep='\t')

        return ftrack, fmodel
    except ValueError as e:
        logger.error(f"Aborting due to error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to process file {praw}: {e}")
        logger.error(traceback.format_exc())
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
    logger = logging.getLogger(__name__)
    try:
        return process_file(*args)
    except ValueError as e:
        logger.error(f"Critical error: {e}. Aborting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}.")
        return None, None

def merge_csv_files(file_pattern, output_file, out_dir):
    logger = logging.getLogger(__name__)
    files = list(out_dir.glob(file_pattern))
    if files:
        df_list = []
        for f in files:
            try:
                df = pd.read_csv(f, compression='gzip', sep='\t')
                if not df.empty:
                    df_list.append(df)
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty file skipped: {f}")
        if df_list:
            df_merged = pd.concat(df_list, ignore_index=True)
            df_merged.to_csv(out_dir / output_file, compression='gzip', index=False, sep='\t')
            logger.info(f"Merged {len(df_list)} files into {output_file}")
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
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-retention", type=int, default=7,
                        help="Número de archivos de log antiguos a conservar")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strdate = time.strftime("%d%m%Y_%H%M")
    flog = out_dir / f'{strdate}.log'

    # Configura el logging
    log_queue, listener = configure_logging(flog, log_level=args.log_level, backup_count=args.log_retention)

    # Usa logging en lugar de print o print-like outputs
    logger = logging.getLogger(__name__)
    logger.info("Starting the main process.")
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Start time: {t0}")
    if args.info:
        logger.info(f"Additional info: {args.info}")  # Información adicional si se proporciona

    # Start profiling
    pr = None
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    # Parámetros de RANSAC
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

    logger.info(f"Number of files to be read: {len(rawdata_path)}")
    total_files = len(rawdata_path)
    start_time = time.time()

    # Proceso paralelo con ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_wrapper, (praw, args, kwargs_ransac, out_dir, n))
                for n, praw in enumerate(rawdata_path)
            ]
            
            # Muestra el progreso del procesamiento
            for n, future in enumerate(as_completed(futures)):
                try:
                    ftrack, fmodel = future.result()
                    if (n + 1) % max(1, total_files // 100) == 0:
                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / (n + 1)) * (total_files - (n + 1))
                        logger.info(f"Progress: {((n + 1) / total_files) * 100:.2f}% - Estimated time remaining: {remaining_time / 60:.2f} minutes")
                except ValueError as e:
                    logger.error(f"Execution aborted due to error: {e}")
                    listener.stop()  # Stop logging
                    sys.exit(1)  # Immediate exit on error
                except Exception as e:
                    logger.error(f"Unexpected critical error: {e}")
                    listener.stop()  # Stop logging
                    sys.exit(1)  # Immediate exit on error
    except ValueError as e:
        logger.error(f"Execution aborted: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected critical error: {e}")
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

    logger.info(t_end)
    logger.info(f"Output directory : {out_dir}")

    # Stop the QueueListener to ensure logs are flushed to the file
    listener.stop()
