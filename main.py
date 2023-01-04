import argparse

import os
import pandas as pd
from time import time
import multiprocessing as mp
from utils import *
import logging
from pathlib import Path



if __name__ == "__main__": # correct syntax?

    start = time() #add - display prgress nicely
    # complete for nice helper
    # parser = argparse.ArgumentParser()
    # parser.parse_args()
    dataset_name = 'Visium_Human_Breast_Cancer'
    data_path = '/Users/efi/Documents/Master in Computer Science/Thesis/microRNA/validation_data/visium' #spatial: inner folder name like dataset name, inside 'spatial' and 'filtered_feature_bc_matrix' folders
    #maybe split to data_path_sc and data_path_spatial
    #spatial files should be unarchived
    results_path = None #optional
    species = 'homo_sapiens' #'mus_musculus' 'homo_sapiens' #optional
    # miR_list = ['mmu-miR-466i-5p', 'mmu-let-7b-5p', 'mmu-miR-505-3p']   #'all' #optional for compute and figures, can be a list of mirs
    miR_list = ['hsa-miR-300', 'hsa-miR-6502-5p', 'hsa-miR-6727-3p']
    cpus = None #optional
    thresh = 0.00001 #optional
    miR_figures = 'all' #optional. default: top_10. if miR_list provided - disregarding this flag. options: 'all', 'top_10', 'bottom_10'
    verbose = False #optional

    mti_data = mti_loader(species)
    data_type = check_data_type(data_path, dataset_name)
    print('Data type detected: ' + data_type)

    if data_type == 'spatial':
        counts = visium_loader(dataset_name, data_path)
        spatial_coors = get_spatial_coors(dataset_name, data_path, counts)
        spots = spatial_coors.shape[0]
    else:
        counts = scRNAseq_loader(dataset_name, data_path)

    counts_norm = normalize_counts(counts)

    if results_path:
        results_path = os.path.join(results_path, dataset_name)
        activity_maps_path = os.path.join(results_path, 'activity_maps')
    else:
        results_path = os.path.join(data_path, 'results', dataset_name) #need to verify
        # results_path = os.path.join(*data_path.split('/')[:-1], 'results', dataset_name) #need to verify
        activity_maps_path = os.path.join(results_path, 'activity_maps')

    if miR_list == 'all': #is that the right way to do it? it changes the flag into a list
        miR_list = list(set(mti_data["miRNA"]))
    else:
        miR_list_figures = miR_list
    print(len(miR_list) ,'microRNAs detected') #verbose

    if not cpus:
        cpus = mp.cpu_count()
    miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs = compute_mir_activity(counts_norm, miR_list, mti_data, results_path, cpus)

    print('Computation time:', (time() - start)//60, 'minutes')
    
    if data_type == 'spatial':
        print('Generating activity map figures')
        if miR_figures == 'top_10':
            mir_activity = sort_activity_spatial(miR_activity_pvals, thresh, spots, results_path, dataset_name)
            miR_list_figures = mir_activity[0,:10]
        elif miR_figures == 'all':
            miR_list_figures = miR_list
        else: #miR_figures == 'bottom_10'
            mir_activity = sort_activity_spatial(miR_activity_pvals, thresh, spots, results_path, dataset_name)
            miR_list_figures = mir_activity[0,-10:]
        produce_spatial_maps(miR_list_figures, miR_activity_pvals, spatial_coors, results_path, dataset_name)






    
    





