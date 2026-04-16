#!/usr/bin/python3
# -*- coding: utf-8 -*-
#package module(s)
from cli.common_args import set_common_parser, load_args_cache, save_args_cache

def set_processing_parser():
    saved_args = load_args_cache()
    parser = set_common_parser()
    parser.add_argument('--nfiles', '-nf', 
                        default=saved_args.get("nfiles", 0), 
                        help="Maximum number of files to process; use 0 for all available files",
                        type=int)
    parser.add_argument('--entry_start', '-e0', 
                        default=saved_args.get("entry_start", 0), 
                        help="Number of events to skip", 
                        type=int)
    parser.add_argument('--nevents', '-nev', 
                        default=saved_args.get("nevents", int(1e4)), 
                        help="Max number of events to reconstruct", 
                        type=int)
    parser.add_argument('--ndisplays', '-nd', 
                        default=saved_args.get("ndisplays", 0), 
                        help="Number of events to display", 
                        type=int)
    parser.add_argument('--adc_calibration', '-ac', 
                        help="Whether to perform ADC calibration (1) or not (0)", 
                        default=saved_args.get("adc_calibration", 1), type=int,
                        # action='store_true',
                        )
    parser.add_argument(
        '--tracking_type', 
        default=saved_args.get("tracking_type", "ransac"),
        choices=['ransac', 'hough', 'other'],
        help="Tracking type: ransac, hough, or other"
    )
    _args, _ = parser.parse_known_args()
    if _args.tracking_type=="ransac":
        set_ransac_args(parser, saved_args)
    else: 
        #Edit new tracking method here
        pass
    ##MULTI PROCESSING
    # parser.add_argument('--nworker', '-w', default=saved_args.get("nworker", 12), help="Number of workers for multiprocessing", type=int)
    # parser.add_argument('--index_subfiles', '-ix', nargs="+",  default=saved_args.get("index_subfiles", []), help="Index subfiles to process in '*_sub<ix>.root'", type=int)
    # parser.add_argument('--memory_max', '-ram', default=saved_args.get("memory_max", 400.), help="Max RAM in GB", type=float)
    return parser

def set_ransac_args(parser, saved_args={}):
    parser.add_argument('--residual_threshold', '-rt', 
                        default=saved_args.get("residual_threshold", 50), 
                        help="Distance-to-model parameter in pixel unit",
                        type=float)
    parser.add_argument('--min_samples', '-ms', 
                        default=saved_args.get("min_samples", 2), 
                        help='Size of the initial sample',
                        type=int)
    parser.add_argument('--max_trials', '-mt', 
                        default=saved_args.get("max_trials", 30), 
                        help='Number of iterations',
                        type=int)    

def get_processing_args() :
    parser = set_processing_parser()
    args = parser.parse_args()
    save_args_cache(vars(args))
    return args

def set_modeling_parser():
    saved_args = load_args_cache()
    parser = set_common_parser()
    parser.add_argument('--voxel_size', '-v', default=saved_args.get("voxel_size", 50), help="Voxel size in mm", type=int)
    return parser

def get_modeling_args() :
    parser = set_processing_parser()
    args = parser.parse_args()
    save_args_cache(vars(args))
    return args