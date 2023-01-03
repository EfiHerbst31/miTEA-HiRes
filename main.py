import argparse

import os
import pandas as pd
from time import time
import multiprocessing as mp
from utils import *


if __name__ == "__main__": # correct syntax?

    start = time() #add - display prgress nicely
    # complete for nice helper
    # parser = argparse.ArgumentParser()
    # parser.parse_args()
    dataset_name = 'validation'
    data_path = '/Users/efi/Documents/Master in Computer Science/Thesis/microRNA/validation_data/singleCell'
    #results_path #optional
    species = 'homo_sapiens' #optional
    #miR_list #optional for compute and figures
    #cpus #optional
    #thresh #optional
    miR_figures = 'top_10' #optional. default: top_10. if miR_list provided - disregarding this flag. options: 'all', 'top_10', 'bottom_10'
    verbose = False #optional

    mti_data = mti_loader(species)
    data_type = check_data_type(data_path, dataset_name)

    if data_type == 'spatial':
        counts = visium_loader(dataset_name, data_path)
        spatial_coors = get_spatial_coors(dataset_name, data_path)
        spots = spatial_coors.shape[0]
    else:
        counts = scRNAseq_loader(dataset_name, data_path)

    counts_norm = normalize_counts(counts)

    if results_path:
        results_path = os.path.join(results_path, dataset_name)
    else:
        results_path = os.path.join(*data_path.split('/')[:-1], dataset_name) #need to verify
        activity_maps_path = os.path.join(results_path, 'activity_maps')

    if not miR_list:
        miR_list = list(set(mti_data["miRNA"]))
    else:
        miR_list_figures = miR_list
    print(len(miR_list) ,'microRNAs detected') #verbose

    if not cpus:
        cpus = mp.cpu_count()
    miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs = compute_mir_activity(counts, miR_list, mti_data, results_path, cpus)

    print('Computation time:', (time() - start)//60, 'minutes')
    
    if data_type == 'spatial':
        print('Generating activity map figures')
        if not thresh:
            thresh = 0.00001
        if miR_figures == 'top_10':
            mir_activity = sort_activity_spatial(miR_activity_pvals, thresh, spots, results_path)
            miR_list_figures = mir_activity[0,:10]
        elif miR_figures == 'all':
            miR_list_figures = miR_list
        else: #miR_figures == 'bottom_10'
            mir_activity = sort_activity_spatial(miR_activity_pvals, thresh, spots, results_path)
            miR_list_figures = mir_activity[0,-10:]
        produce_spatial_maps(miR_list_figures, miR_activity_pvals, spatial_coors, results_path, dataset_name)






    
    





