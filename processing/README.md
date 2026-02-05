## 1. Processing:  
To process raw telescope data and reconstruct particle track event-by-event, the user needs to run ```python3 processing/main.py``` with the following arguments:    
INPUTS:  
    - ```--telescope [-tel]``` (str2telescope) the telescope name (required): Check the available telescope configurations in  dictionary ```dict_tel``` in ```telescope/telescope.py```.  
    - ```--run [-r]``` (str) which telescope run (e.g.```tomo```, ```calib```), check ```survey/survey.yaml```
    - (optional) ```--max_nfiles [-nf]```  (int, default is ```1```) the maximum number of data files to process.  
    - (optional) ```--max_nevents [-nev]```  (int, default is ```1```) the maximum number of events to process.  
    RANSAC parameters:  
    - (optional) ```--residual_threshold [-rt]```  (float) ransac distance criteria in detector pixel unit
    - (optional) ```--min_samples [-ms]```  (int) the size of the inital data sample  
    - (optional) ```--max_trials [-mt]```  (int) maximum number of iterations to find best trajectory model  
    - ...  

OUTPUTS:  
    Dataframe```df_track.csv.gz```  