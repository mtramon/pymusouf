### Prerequisite
Package setup, check ```INSTALL```.

### Reconstruction: Muon Tracking
To process raw telescope data and reconstruct particle track event-by-event, the user needs to run ```python3 processing/tracks.py``` with the following arguments:    
INPUTS:  
    - ```--telescope [-tel]``` (str2telescope) the telescope name (required): Check the available telescope configurations in ```survey/survey.yaml```or in  dictionary ```DICT_TEL``` in ```telescope/telescope.py```.  
    - ```--run [-r]``` (str) which telescope run (e.g.```tomo```, ```calib```), check ```survey/survey.yaml```
    - (optional) ```--max_nfiles [-nf]```  (int, default is ```1```) the maximum number of data files to process.  
    - (optional) ```--max_nevents [-nev]```  (int, default is ```1e4```) the maximum number of events to process.  
    RANSAC parameters:  
    - (optional) ```--residual_threshold [-rt]```  (float,  default is ```1```) ransac distance criteria in detector pixel unit
    - (optional) ```--min_samples [-ms]```  (int,  default is ```2```) the size of the inital data sample  
    - (optional) ```--max_trials [-mt]```  (int,  default is ```30```) maximum number of iterations to find best trajectory model  
    - ...  

OUTPUT:  
    A dataframe ```df_track.csv.gz``` with columns ```["event_id","timestamp","npts", "config", "dx_dz","dy_dz","dz", "inside","rms","use_ransac"]```. The file contains the reconstructed tracks per event index in each panel configuration ```"config"``` per line (```"3p1"``` only or ```["3p1","3p2","4p"]```).

**Note:** During reconstruction, the script creates a .args_cache.json file in the current working directory. It stores parsed arguments and reuses them as defaults for subsequent runs.

See ```cli/common_args.py``` and ```cli/processing_args.py``` for further details on available options.

---

### Event rate and brut images
See ```plot_images.py```.

Check reconstructed event rate and 2d histograms (```dy_dz``` vs ```dx_dz```).

---

### Pipeline to opacity estimates
See ```muography.py```.

Save multiple images and rays travel length in volume of interest in ```muography.h5``` file.

