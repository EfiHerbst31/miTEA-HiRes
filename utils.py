import csv
import glob
import multiprocessing as mp
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import xlmhg


def check_data_type(data_path: str, dataset_name: str) -> str:
    """Checks if data is single cell or spatial.
    
    If 'filtered_feature_bc_matrix' exists in the expected folder, 
    data type is considered spatial, and scRNAseq otherwise.

    Args:
        dataset_name: dataset folder name.
        data_path: path for dataset folder.

    Returns:
        the data type of the provided dataset, "scRNAseq" or "spatial".
    """

    dataset_path = os.path.join(data_path, dataset_name)
    counts_path = os.path.join(dataset_path, "filtered_feature_bc_matrix")
    is_spatial = os.path.exists(counts_path)
    if is_spatial:
        return 'spatial'
    else:
        return 'scRNAseq'


def detect_data_file(data_path: str, dataset_name: str) -> str:
    """Checks scRNAseq file extention and returns detected file name.
    
    Supported extensions: txt, tsv or pkl.
    Prioratizing for pkl file, assuming it contains processed data.

    Args:
        dataset_name: dataset folder name.
        data_path: path for dataset folder.

    Return:
        the detected file name.

    Raises:    
        Exception if more than one file was detected.
        Exception if no file of types txt, tsv or pkl was found.
    """

    pkl_file = glob.glob(data_path + '/*.pkl')
    txt_files = glob.glob(data_path + '/*.txt')
    tsv_files = glob.glob(data_path + '/*.tsv')
    if pkl_file:
        if len(pkl_file) > 1: 
            print("error: there seem to be more than one pkl file, \
            please merge to a single file") #proper error
        else:
            file_name = pkl_file[0]        
    elif txt_files:
        if len(txt_files) > 1: 
            print("error: there seem to be more than one txt file, \
                please merge to a single file") #proper error
        else:
            file_name = txt_files[0]
    elif tsv_files:
        if len(tsv_files) > 1: 
            print("error: there seem to be more than one tsv file, \
                please merge to a single file") #proper error
        else:
            file_name = tsv_files[0]
    try:
        return file_name #need to fix - 'UnboundLocalError: local variable 'file_name' referenced before assignment'
    except:
        raise Exception("error: no pkl, txt or tsv data files \
            were detected in: ", data_path) #proper error
    

def mti_loader(species: str = 'homo_sapiens') -> pd.DataFrame:
    """Loads MTI data.
    
    If other species is required, please download the mti data and 
        update this function.

    Args:
        species: 'homo_sapiens' (default) or 'mus_musculus'.

    Returns:
        MTI (microRNA targets) data.
    
    Raises:
        exception if species type is not recognized.
    """

    mti_loc = 'miRTarBase/release_8.0'
    mti_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), mti_loc)
    print(mti_path)
    homo_mti_file = 'hsa_MTI_filtered.csv'
    mus_mti_file = 'mmu_MTI_filtered.csv'

    print('Loading MTIs...') #change to verbose
    if species == "homo_sapiens":
        mti_data = pd.read_csv(os.path.join(mti_path, homo_mti_file), index_col=0)
    elif species == "mus_musculus":
        mti_data = pd.read_csv(os.path.join(mti_path, mus_mti_file), index_col=0)
    else:
        raise Exception('type of species not recognized, either "homo_sapiens" or \
            "mus_musculus" are supported. For other species type, \
                please download the mti data file and update mti_loader function')
    return mti_data 


def switch_10x_to_txt(matrix_mtx_file: str, features_tsv_file: str, 
    barcodes_tsv_file: str, save_to_file: bool = False, 
    path_to_save: str = None) -> pd.DataFrame:
    """Converts visium data to reads table 
        where columns are the spots and rows are gene reads.

    Args:
        matrix_mtx_file: path to matrix.mtx that was downloaded from visium website.
        features_tsv_file: path to features.tsv that was downloaded from visium website.
        barcodes_tsv_file: path to barcodes.tsv that was downloaded from visium website.
        save_to_file: save generated data table to file at new_txt_file, default = False.
        path_to_save: path to save the generated data table if save_to_file is True, default = None.

    Returns:
        reads table.
    """

    the_matrix = scipy.io.mmread(matrix_mtx_file).todense()

    with open(features_tsv_file) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        genes = []
        for row in rd:
            genes.append(row[1])

    with open(barcodes_tsv_file) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        cells = []
        for row in rd:
            cells.append(row[0])

    data = pd.DataFrame(the_matrix, columns=cells, index=genes)

    if save_to_file:
        data.to_csv(path_to_save, sep='\t')

    return data


def visium_loader(dataset_name: str, data_path: str) -> pd.DataFrame:
    """Loads visium data.

    Args:
        dataset_name: dataset folder name.
        data_path: path to dataset folder.
    
    Returns:
        reads table with spots (columns) and genes (rows).
    """

    print('Loading visium data...') #change to verbose
    dataset_path = os.path.join(data_path, dataset_name)
    counts_path = os.path.join(dataset_path, "filtered_feature_bc_matrix")
    matrix_mtx_file = os.path.join(counts_path, "matrix.mtx")
    features_tsv_file = os.path.join(counts_path, "features.tsv")
    barcodes_tsv_file = os.path.join(counts_path, "barcodes.tsv")
    counts = switch_10x_to_txt(matrix_mtx_file, features_tsv_file, barcodes_tsv_file)

    return counts


def get_spatial_coors(dataset_name: str, data_path: str, counts: pd.DataFrame) \
     -> pd.DataFrame:
    """Loads spatial coordinates.

    Args:
        dataset_name: dataset folder name.
        data_path: path to dataset folder.
        counts: reads table.

    Returns:
        spatial coordinates.
    """

    dataset_path = os.path.join(data_path, dataset_name)
    spatial_path = os.path.join(dataset_path, "spatial")
    try:
        spatial_coors = pd.read_csv(os.path.join(spatial_path, "tissue_positions_list.csv"), 
            index_col=0, sep=',', header='infer')
        spatial_coors = spatial_coors.loc[list(counts)]
        spatial_coors = np.array(spatial_coors[['array_col','array_row']])
    except:
        try:
            spatial_coors = pd.read_csv(os.path.join(spatial_path, "tissue_positions_list.csv"), 
                index_col=0, sep=',',header=None) #verify different versions
            spatial_coors = spatial_coors.loc[list(counts)]
            spatial_coors = np.array(spatial_coors[[3, 2]])
        except:
            spatial_coors = pd.read_csv(os.path.join(spatial_path, "tissue_positions.csv"), 
                index_col=0, sep=',', header='infer')
            spatial_coors = spatial_coors.loc[list(counts)]
            spatial_coors = np.array(spatial_coors[['array_col','array_row']])

    return spatial_coors


def scRNAseq_loader(dataset_name: str, data_path: str) -> pd.DataFrame:
    """Loads scRNAseq data.

    Args:
        dataset_name: dataset folder name.
        data_path: path to dataset folder.

    Returns:
        reads table with cells (columns) and genes (rows).
    """

    print('Loading scRNAseq data...') #change to verbose
    dataset_path = os.path.join(data_path, dataset_name)
    file_name = detect_data_file(data_path, dataset_name)
    reads_file = os.path.join(dataset_path, file_name)
    if file_name.endswith('.txt'):
        counts = pd.read_csv(reads_file, delimiter="\t", index_col=0)        
    elif file_name.endswith('.tsv'):
        counts = pd.read_csv(reads_file, sep='\t', index_col=0, on_bad_lines='skip').T #need to verify
    else: #pkl file
        counts = pd.read_pickle(reads_file)
    return counts


def normalize_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Normalizing reads table.

    Removing genes (rows) where all reads are zero.
    Normalizing every spot/cell (column) such that 
        they have the same amount of total reads.
    z-score transformation per gene (row).
    
    Args: 
        counts: reads table.

    Returns: 
        normalized reads table.
    """

    counts_norm = counts.loc[counts.sum(axis=1) > 0]
    counts_norm = counts_norm.divide(counts_norm.sum(), axis='columns').\
        multiply(10000)
    counts_norm = counts_norm.subtract(counts_norm.mean(axis=1), axis='index').\
        divide(counts_norm.std(axis=1), axis='index')
    return counts_norm


def compute_mir_activity(counts: pd.DataFrame, miR_list: list, mti_data: pd.DataFrame, 
    results_path: str, cpus: int) -> pd.DataFrame:
    """Computing microRNA activity.

    Multiprocessing of per cell/spot per microRNA computations.
    Using mHG test.

    Args:
        counts: reads table.
        miR_list: microRNAs to consider.
        mti_data: microRNA targets data.
        results_path: path to save results.
        cpus: amount of cpus to use in parallel.

    Returns: 
        microRNA activity results achived using mHG test.
    """

    miR_activity_stats = pd.DataFrame(columns=list(counts), index=miR_list)
    miR_activity_pvals = pd.DataFrame(columns=list(counts), index=miR_list)
    miR_activity_cutoffs = pd.DataFrame(columns=list(counts), index=miR_list)

    print('Computing activity map...') #verbose?

    with mp.Pool(cpus) as pool:
        result_list = pool.starmap(compute_stats_per_cell,
                                   [(cell, counts.loc[:, cell].sort_values(),
                                     miR_list, mti_data) for cell in list(counts)])

    for result in result_list:
        cell = result[0]
        miR_activity_stats.loc[:, cell] = result[1]
        miR_activity_pvals.loc[:, cell] = result[2]
        miR_activity_cutoffs.loc[:, cell] = result[3]

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    miR_activity_stats.to_csv(results_path + '/activity_stats.csv')
    miR_activity_pvals.to_csv(results_path + '/activity_pvals.csv')
    miR_activity_cutoffs.to_csv(results_path + '/activity_cutoffs.csv')

    return miR_activity_pvals


def compute_stats_per_cell(cell: str, ranked: pd.DataFrame, miR_list: list, 
    mti_data: pd.DataFrame) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computing microRNA activity per spot/cell.

    Using mHG test.
    If no targets are found in the reads table for a particular microRNA, 
    the result will be p-value of 0 for all cells/spots.

    Args:
        cell:  column id.
        ranked: sorted reads column for the cell.
        miR_list: microRNAs to consider.
        mti_data: microRNA targets data.

    Returns: 
        column id and microRNA activity results achived using mHG test.
    """

    # Parameters for mHG package
    X = 1
    L = len(ranked)

    # if verbose:
    #     print(cell)

    ranked_list = list(ranked.index)
    miR_activity_stats = []
    miR_activity_pvals = []
    miR_activity_cutoffs = []
    for miR in miR_list:
        miR_targets = list(mti_data[mti_data["miRNA"] == miR]["Target Gene"])
        v = np.uint8([int(g in miR_targets) for g in ranked_list])
        # if sum(v) == 0: #somehow add to log if all are 1 for this microRNA
        #     print('no targets found for ' + miR)
        #     print(miR_targets)
        #     raise Exception('no targets were found for microRNA: ' + miR + '. please check that correct \'species\' flag was selected')
        stat, cutoff, pval = xlmhg.xlmhg_test(v, X=X, L=L)
        miR_activity_stats.append(stat)
        miR_activity_cutoffs.append(cutoff)
        miR_activity_pvals.append(pval)

    return cell, miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs


def sort_activity_spatial(miR_activity_pvals: pd.DataFrame, thresh: float, 
    spots: int, results_path: str, dataset_name: str) -> pd.DataFrame:
    """Computes which microRNAs are the most active within the entire slide.

    Args:
        miR_activity_pvals: activity table per spot per microRNA.
        thresh: used for filtering only microRNAs/spt that got a score lower than thresh, 
            i.e. very active.
        spots: number of spots.
        results_path: where to save the sorted list of most highly expressed microRNA.
    
    Returns: 
        sorted list of microRNA, from the most overall active to the least, 
            over the extire slide.
    """

    mir_expression = miR_activity_pvals[miR_activity_pvals < thresh].\
        count(axis=1).sort_values(ascending = False)
    mir_expression = mir_expression / spots
    mir_expression.to_csv(results_path + '/mir_expression_th' + str(thresh) \
        + '_' + dataset_name + '.csv')

    return mir_expression


def produce_spatial_maps(miR_list_figures: list, miR_activity_pvals: pd.DataFrame, 
    spatial_coors: pd.DataFrame, results_path: str, dataset_name: str):
    """Produces a figure with activity map per microRNA in the list.

    Args:
        miR_list_figures: list of microRNAs to produce figures for.
        miR_activity_pvals: activity per spot per microRNA.
        spatial_coors: spatial location of each spot.
        results_path: path to save figures.
        dataset_name: for plot name.
    
    Returns:
        None
    """

    results_path_figures = os.path.join(results_path, 'activity maps')
    if not os.path.exists(results_path_figures):
        os.makedirs(results_path_figures)

    for miR in miR_list_figures:
        pvals = miR_activity_pvals.loc[miR, :]
        log10_pvals = -np.log10(pvals)
        path_to_plot = results_path_figures + '/' + dataset_name + '_' + miR + '.jpg'
        plt.figure(figsize = (10, 10))
        plt.scatter(spatial_coors[:, 0], spatial_coors[:, 1], c=log10_pvals, 
            vmin=np.min(log10_pvals), vmax=np.max(log10_pvals))
        plt.gca().invert_yaxis()
        plt.colorbar(extend='max').set_label('p-value (-log10)', rotation=270, labelpad=20)
        plt.title(miR + ' activity map', fontsize=14)
        plt.savefig(path_to_plot)



# preprocessing functions