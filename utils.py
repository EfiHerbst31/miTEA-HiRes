import xlmhg
import csv
import scipy.io
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import glob
import matplotlib as plt


def check_data_type(data_path, dataset_name):
    """
    checks if single cell or spatial data
    
    input:
    param dataset_name: dataset folder name
    param data_path: path for dataset folder

    return:
    data_type - "scRNAseq" or "spatial"
    """
    dataset_path = os.path.join(data_path, dataset_name)
    counts_path = os.path.join(dataset_path, "filtered_feature_bc_matrix")
    is_spatial = os.path.isdir(counts_path)
    if is_spatial:
        data_type = 'spatial'
    else:
        data_type = 'scRNAseq'

    return data_type

def detect_data_file(data_path, dataset_name):
    """
    checks scRNAseq file extention and returns detected file name
    
    input:
    param dataset_name: dataset folder name
    param data_path: path for dataset folder

    return:
    file_name - detected file, either txt or tsv are supported, single file
    """
    txt_files = glob.glob(data_path + '/*.txt')
    tsv_files = glob.glob(data_path + '/*.tsv')
    pkl_file = glob.glob(data_path + '/*.pkl')
    if pkl_file:
        if len(pkl_file) > 1: 
            print("error: there seem to be more than one pkl file, please merge to a single file") #proper error
        else:
            file_name = pkl_file[0]        
    elif txt_files:
        if len(txt_files) > 1: 
            print("error: there seem to be more than one txt file, please merge to a single file") #proper error
        else:
            file_name = txt_files[0]
    elif tsv_files:
        if len(tsv_files) > 1: 
            print("error: there seem to be more than one tsv file, please merge to a single file") #proper error
        else:
            file_name = tsv_files[0]
    try:
        return file_name #need to fix - 'UnboundLocalError: local variable 'file_name' referenced before assignment'
    except:
        raise ValueError("error: no pkl, txt or tsv data files were detected in: ", data_path) #proper error
    


def mti_loader(species = 'homo_sapiens'):
    """
     Loads mti data
     param species (optional): 'homo_sapiens' (default) or 'mus_musculus'
    """

    mti_loc = 'miRTarBase/release_8.0'
#    print(os.path.dirname(os.path.realpath(__file__)))
    #mti_path = os.path.join(os.curdir(__file__),mti_loc)
    mti_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), mti_loc)
    print(mti_path)
    homo_mti_file = 'hsa_MTI_filtered.csv'
    mus_mti_file = 'mmu_MTI_filtered.csv'

    print('Loading MTIs...') #change to verbose
    if species == "homo_sapiens":
        mti_data = pd.read_csv(os.path.join(mti_path, homo_mti_file), index_col=0)
    else:
        mti_data = pd.read_csv(os.path.join(mti_path, mus_mti_file), index_col=0)

    return mti_data #need to return this?


def switch_10x_to_txt(matrix_mtx_file, features_tsv_file, barcodes_tsv_file, save_to_file=False, path_to_save=None):
    """
    converts visium data to reads pandas table where columns are the spots and rows are gene reads.
    param: matrix_mtx_file - as downloaded from visium
    param: features_tsv_file - as downloaded from visium
    param: barcodes_tsv_file - as downloaded from visium
    param (optional): save_to_file - save generated data table to file at new_txt_file, default = False.
    param (optional): path_to_save - path to save the generated data table if save_to_file is True, default = None
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


def visium_loader(dataset_name, data_path):
    """
     Loads visium data
     input:
        param dataset_name: dataset folder name
        param data_path: path for dataset folder
    return counts: reads table with spots (columns) and genes (rows)
    """

    print('Loading visium data...') #change to verbose
    dataset_path = os.path.join(data_path, dataset_name)
    counts_path = os.path.join(dataset_path, "filtered_feature_bc_matrix")
    matrix_mtx_file = os.path.join(counts_path, "matrix.mtx")
    features_tsv_file = os.path.join(counts_path, "features.tsv")
    barcodes_tsv_file = os.path.join(counts_path, "barcodes.tsv")
    counts = switch_10x_to_txt(matrix_mtx_file, features_tsv_file, barcodes_tsv_file)

    return counts

def get_spatial_coors(dataset_name, data_path):
    """
    Loads spatial coordinates
    input:
        param dataset_name: dataset folder name
        param data_path: path for dataset folder
    return spatial_coors
    """

    dataset_path = os.path.join(data_path, dataset_name)
    spatial_path = os.path.join(dataset_path, "spatial")
    spatial_coors = pd.read_csv(os.path.join(spatial_path, "tissue_positions_list.csv"), index_col=0, sep=',',header=None) #verify different versions
    spatial_coors = spatial_coors.loc[list(counts)]
    spatial_coors = np.array(spatial_coors[[3, 2]])
    return spatial_coors

def scRNAseq_loader(dataset_name, data_path):
    """
    Loads scRNAseq data
    input:
        param dataset_name: dataset folder name
        param data_path: path for dataset folder
    return counts: reads table with cells (columns) and genes (rows)
    """
    print('Loading scRNAseq data...') #change to verbose
    dataset_path = os.path.join(data_path, dataset_name)
    file_name = detect_data_file(data_path, dataset_name)
    reads_file = os.path.join(dataset_path, file_name)
    if file_name.endswith('.txt'):
        counts = pd.read_csv(reads_file, delimiter="\t", index_col=0)        
    elif file_name.endswith('.tsv'):
        counts = pd.read_csv(reads_file, sep='\t', index_col=0, on_bad_lines='skip').T #need to verify
    else: #pkl
        counts = pd.read_pickle(reads_file)
    return counts

def normalize_counts(counts):
    """
    removing genes (rows) where all reads are zero
    normalizing every spot/cell (column) such that they have the same amount of total reads
    z-score transformation per gene (row)
    
    param: counts - reads table
    return: norm_counts - normalized reads table
    """

    counts_norm = counts.loc[counts.sum(axis=1) > 0]
    counts_norm = counts_norm.divide(counts_norm.sum(), axis='columns').multiply(10000)
    counts_norm = counts_norm.subtract(counts_norm.mean(axis=1), axis='index').divide(counts_norm.std(axis=1), axis='index')
    return counts_norm

def compute_mir_activity(counts, miR_list, mti_data, results_path, cpus):
    """
    computing microRNA activity
    input:
        param: counts - reads table
        param: miR_list - microRNAs to consider
        param: mti_data - microRNA targets data
        param: results_path - path to save results
        param: cpus - amount of 
    return: miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs - miR activity results
    """
    #initializing results tables
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
        os.mkdir(results_path)
    miR_activity_stats.to_csv(results_path + '/activity_stats.csv')
    miR_activity_pvals.to_csv(results_path + '/activity_pvals.csv')
    miR_activity_cutoffs.to_csv(results_path + '/activity_cutoffs.csv')

    return miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs

def compute_stats_per_cell(cell, ranked, miR_list, mti_data):
    """
    computing microRNA activity per spot/cell
    input:
        param: cell -  column id
        param: ranked - sorted reads column for the cell
        param: miR_list - microRNAs to consider
        param: mti_data - microRNA targets data
    return: 
        cell, miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs
    """

    #setting parameters for mHG package
    X = 1
    L = len(counts)

    if verbose:
        print(cell)
    
    ranked_list = list(ranked.index)
    miR_activity_stats = []
    miR_activity_pvals = []
    miR_activity_cutoffs = []
    for miR in miR_list:
        miR_targets = list(mti_data[mti_data["miRNA"] == miR]["Target Gene"])
        v = np.uint8([int(g in miR_targets) for g in ranked_list])
        stat, cutoff, pval = xlmhg.xlmhg_test(v, X=X, L=L)
        miR_activity_stats.append(stat)
        miR_activity_cutoffs.append(cutoff)
        miR_activity_pvals.append(pval)

    return (cell , miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs)

def sort_activity_spatial(miR_activity_pvals, thresh, spots, results_path):
    """
    computes which microRNAs are the most active within the entire slide
    input:
        param: miR_activity_pvals - activity table per spot per microRNA 
        param: thresh - used for filtering only microRNAs/spt that got a score lower than thresh, i.e. very active
        param: spots - number of spots
        param: results_path - where to save the sorted list of most highly expressed microRNA
    return: mir_expression - sorted list of microRNA, from the most overall active to the least, over the extire slide
    """
    expressed_mir = miR_activity_pvals[miR_activity_pvals < thresh].count(axis=1).sort_values(ascending = False)
    expressed_mir = expressed_mir / spots
    expressed_mir.to_csv(results_path + '/expressed_mir_th' + str(thresh) + '_' + dataset_name + '.csv')

    return mir_expression

def produce_spatial_maps(miR_list_figures, miR_activity_pvals, spatial_coors, results_path, dataset_name):
    """
    produces a figure with activity map per microRNA in the list
    input:
        param: miR_list_figures - list of microRNAs to produce figures for
        param: miR_activity_pvals - activity per spot per microRNA
        param: spatial_coors - spatial location of each spot
        param: results_path - path to save figures
        param: dataset_name - for plot name
    """
    results_path_figures = os.path.join(results_path, 'activity maps')
    if not os.path.exists(results_path_figures):
        os.mkdir(results_path_figures)

    for miR in miR_list_figures:
        pvals = miR_activity_pvals.loc[miR, :]
        log10_pvals = -np.log10(pvals)
        path_to_plot = results_path_figures + '/' + dataset_name + '_' + miR + '.jpg'
        plt.figure(figsize = (10, 10))
        plt.scatter(spatial_coors[:, 0], spatial_coors[:, 1], c=log10_pvals, vmin=np.min(log10_pvals), vmax=np.max(log10_pvals))
        plt.gca().invert_yaxis()
        plt.colorbar(ax, extend='max').ax.set_ylabel('p-value (-log10)', rotation=270, labelpad=20)
        plt.title(miR + ' activity map', fontsize=14)
        plt.savefig(path_to_plot)












# preprocessing functions