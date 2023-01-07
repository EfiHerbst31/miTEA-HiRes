#TO DOs:
# add - display prgress nicely
# add optional preprocessing for merging and sampling sc datasets
# remove all comments
# move prints to logs
# add as many errors as neededL what is the problem and how to solve it
# add preprocessing function in utils

#miR_list = ['mmu-miR-466i-5p', 'mmu-let-7b-5p', 'mmu-miR-505-3p']
#miR_list = ['hsa-miR-300', 'hsa-miR-6502-5p', 'hsa-miR-6727-3p']
#--dataset_name='Visium_Human_Breast_Cancer' --data_path='/Users/efi/Documents/Master in Computer Science/Thesis/microRNA/validation_data/visium' --miR_list=hsa-miR-300,hsa-miR-6502-5p,hsa-miR-6727-3p

import logging
import multiprocessing as mp
import os
from pathlib import Path
from time import time

from absl import app
from absl import flags
from absl import logging
import pandas as pd

import utils


FLAGS = flags.FLAGS

flags.DEFINE_integer('cpus', None, 'Number of CPUs to use in parallel. \
default: all available cpus are used.')    
flags.DEFINE_string('data_path', None, 'Path to dataset. \
if scRNAseq: path to dataset. if spatial: path to the folder containing \
\'spatial\' and \'filtered_feature_bc_matrix\' folders.')
flags.DEFINE_string('dataset_name', None, 'Name of your dataset.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_string('miR_figures', 'top_10', 'Which microRNAs activity maps to draw. \
options: \'all\', \'bottom_10\', \'top_10\'')
flags.DEFINE_list('miR_list', None, 'Comma-separated list of microRNAs to compute. \
default: all microRNAs are computed.\
 Example use: --miR_list=hsa-miR-300,hsa-miR-6502-5p,hsa-miR-6727-3p')
flags.DEFINE_string('results_path', None, 'Path to save results.')
flags.DEFINE_string('species', 'homo_sapiens', 'Options: \'homo_sapiens\' or \
\'mus_musculus\'.')
flags.DEFINE_float('thresh', 0.00001, 'Threshold of the microRNA activity p-value. \
if a microRNA receives a lower score than \'thresh\' it is considered active. \
used in order to find the most active microRNAs across the provided data.')

flags.register_validator('species', 
                         lambda value: value == 'homo_sapiens' or 'mus_musculus',
                         message='Either \'homo_sapiens\' or \'mus_musculus\' are supported.')
flags.register_validator('miR_figures', 
                         lambda value: value == 'top_10' or 'all' or 'bottom_10',
                         message='Either \'top_10\' or \'all\' or \'bottom_10\' are supported.')
      
flags.mark_flag_as_required('dataset_name')
flags.mark_flag_as_required('data_path')

def main(argv):
    if FLAGS.results_path:
        results_path = os.path.join(FLAGS.results_path, FLAGS.dataset_name)
    else:
        results_path = os.path.join(FLAGS.data_path, 'results', FLAGS.dataset_name)
    cpus = FLAGS.cpus or mp.cpu_count()

    mti_data = utils.mti_loader(FLAGS.species)
    miR_list = list(FLAGS.miR_list) or list(set(mti_data["miRNA"]))
    print(len(miR_list) ,'microRNAs detected') 

    data_type = utils.check_data_type(FLAGS.data_path, FLAGS.dataset_name)
    logging.info('Data type detected: %s', data_type)

    if data_type == 'spatial':
        counts = utils.visium_loader(FLAGS.dataset_name, FLAGS.data_path)
        spatial_coors = utils.get_spatial_coors(FLAGS.dataset_name, FLAGS.data_path, counts)
        spots = spatial_coors.shape[0]
    else:
        counts = utils.scRNAseq_loader(FLAGS.dataset_name, FLAGS.data_path)

    counts_norm = utils.normalize_counts(counts)
    
    start = time() 
    miR_activity_pvals = utils.compute_mir_activity(counts_norm, miR_list, mti_data, results_path, cpus)
    print('Computation time:', (time() - start)//60, 'minutes')
    
    if data_type == 'spatial':
        print('Generating activity map figures')
        if FLAGS.miR_figures == 'all' or len(miR_list) <= 10:
            miR_list_figures = miR_list
        elif FLAGS.miR_figures == 'top_10':
            mir_activity = utils.sort_activity_spatial(miR_activity_pvals, FLAGS.thresh, spots, results_path, FLAGS.dataset_name)
            miR_list_figures = mir_activity[0,:10]
        else: #'bottom_10'
            mir_activity = utils.sort_activity_spatial(miR_activity_pvals, FLAGS.thresh, spots, results_path, FLAGS.dataset_name)
            miR_list_figures = mir_activity[0,-10:]
        utils.produce_spatial_maps(miR_list_figures, miR_activity_pvals, spatial_coors, results_path, FLAGS.dataset_name)

if __name__ == '__main__': 
    app.run(main)