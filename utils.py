import csv
import glob
import gzip
import multiprocessing.pool as mp
import os
import shutil
from typing import Optional, Tuple

from absl import logging
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
from scipy.stats import ranksums
import tqdm    
import xlmhg

import constants

_HOMO_MTI_FILE = 'hsa_MTI_filtered.csv'
_MUS_MTI_FILE = 'mmu_MTI_filtered.csv'
_MHG_X_PARAM = 1
_NON_ACTIVE_THRESH = 1.5
_VERY_ACTIVE_THRESH = 7
_10X_SCRNASEQ_FEATURES_SUFFIX = '_genes.tsv'
_10X_SCRNASEQ_BARCODES_SUFFIX = '_barcodes.tsv'


class MyPool(mp.Pool):
    '''Class extending multiprocessing.pool.Pool with starmap method that can work with tqdm.'''
    def istarmap(self, func, iterable, chunksize=1):
        '''Starmap-version of imap.

        Args:
            Same as mp.pool.Pool.starmap.
        
        Returns:
            Same as mp.pool.Pool.starmap.
        '''
        self._check_running()
        if chunksize < 1:
            raise ValueError('Chunksize must be 1+, not {0:n}'.format(chunksize))

        task_batches = mp.Pool._get_tasks(func, iterable, chunksize)
        result = mp.IMapIterator(self)
        self._taskqueue.put(
            (self._guarded_task_generation(
                result._job, mp.starmapstar, task_batches), result._set_length))
        return (item for chunk in result for item in chunk)

class UsageError(Exception):
    '''Error type that is used when there is a problem with package usage.'''
    pass

def check_data_type(data_path: str) -> str:
    '''Checks if data is single cell or spatial.
    
    If 'filtered_feature_bc_matrix' exists in the expected folder, 
    data type is considered spatial, and scRNAseq otherwise.

    Args:
        data_path: Path for dataset folder.

    Returns:
        The data type of the provided dataset, 'scRNAseq' or 'spatial'.
    '''
    counts_path = os.path.join(data_path, constants._SPATIAL_FOLDER_2)
    is_spatial = os.path.exists(counts_path)
    if is_spatial:
        logging.info('Data type detected: %s', constants._DATA_TYPE_SPATIAL)
        return constants._DATA_TYPE_SPATIAL
    else:
        logging.info('Data type detected: %s', constants._DATA_TYPE_SINGLE_CELL)
        return constants._DATA_TYPE_SINGLE_CELL

def get_data_file(data_path: str, file_extension: str) -> Optional[str]:
    '''Checks for existing file with specific extension in the data path.

    Args:
        data_path: path for dataset.
        file_extension: file extension.

    Returns:
        The detected file.

    Raises:    
        UsageError: if more than one file was detected.
    '''
    path_to_files = os.path.join(data_path, '*.%s' %file_extension)
    files = glob.glob(path_to_files)
    if len(files) > 1:
        raise UsageError('There seem to be more than one %s file. '
                         'please merge to a single file' %file_extension)
    return next(iter(files), None)

def detect_data_file(data_path: str) -> str:
    '''Checks scRNAseq file extention and returns detected file name.
    
    Supported extensions: txt, tsv or pkl.
    Prioratizing for pkl file, assuming it contains processed data.

    Args:
        data_path: path for dataset.

    Returns:
        The detected file name.

    Raises:    
        UsageError: if no file of types txt, tsv or pkl was found.
    '''
    data_file = (
        get_data_file(data_path, 'pkl') or
        get_data_file(data_path, 'txt') or
        get_data_file(data_path, 'tsv')
    )

    if not data_file:
        raise UsageError('No \'txt\', \'tsv\' or \'mtx\' files were found in %s. '
                         'Please try passing -preprocess=True' %data_path)

    return data_file
    

def mti_loader(species: str=constants._SPECIES_HOMO_SAPIENS) -> pd.DataFrame:
    '''Loads MTI data.
    
    If other species is required, please download the mti data and update this function.

    Args:
        species: 'homo_sapiens' (default) or 'mus_musculus'.

    Returns:
        MTI (microRNA targets) data.
    
    Raises:
        UsageError: if species type is not recognized.
    '''
    logging.debug('Loading microRNA target data')
    mti_loc = 'miRTarBase/release_8.0'
    mti_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), mti_loc)
    logging.debug('mti data path: %s' %mti_path)
    logging.debug('Loading MTIs...')
    if species == constants._SPECIES_HOMO_SAPIENS:
        homo_file_path = os.path.join(mti_path, _HOMO_MTI_FILE)
        mti_data = pd.read_csv(homo_file_path, index_col=0)
    elif species == constants._SPECIES_MUS_MUSCULUS:
        mus_file_path = os.path.join(mti_path, _MUS_MTI_FILE)
        mti_data = pd.read_csv(mus_file_path, index_col=0)
    else:
        raise UsageError('Type of species not recognized, either %s are supported. '
                         'For other species type, please download the mti data file and '
                         'update mti_loader function' % ' or '.join(constants._SUPPORTED_SPECIES))
    return mti_data 

def mir_data_loading(miR_list: Optional[list]=None, 
    species: Optional[str]=constants._SPECIES_HOMO_SAPIENS, 
    debug: Optional[bool]=False) -> Tuple[pd.DataFrame, list]:
    '''Loading microRNA target data.

    Args:
        miR_list: (optional) list of microRNAs to compute.
        species: (optional) either 'homo_sapiens' (default) or 'mus_musculus' are supported. 
        debug: (optional) if True, provides aditional information. Default=False.

    Returns:
        mti_data: microRNA targets data.
        miR_list: microRNA list to compute.
    
    Raises:
        UsageError if species doesn't match microRNAs provided in the list.
    '''
    mti_data = mti_loader(species=species)
    logging.debug('Loading microRNA list')
    miR_list = list(miR_list) if miR_list else list(set(mti_data['miRNA']))
        
    logging.info('Number of microRNAs detected: %i',len(miR_list))
    logging.debug('MicroRNA list: %s', miR_list)

    if  species == constants._SPECIES_HOMO_SAPIENS and \
        constants._HOMO_SAPIENS_PREFIX not in miR_list[0]:
        raise UsageError('species is %s but some of the microRNAs in the list do not '
                               'belong to humans', constants._SPECIES_HOMO_SAPIENS)
    elif species == constants._SPECIES_MUS_MUSCULUS and \
        constants._MUS_MUSCULUS_PREFIX not in miR_list[0]:
        raise UsageError('species is %s but some of the microRNAs in the list do not '
                               'belong to mice', constants._SPECIES_MUS_MUSCULUS)
    else:
        logging.debug('Species match microRNAs in the list')

    return mti_data, miR_list

def switch_10x_to_txt_sc(matrix_mtx_file: str, features_tsv_file: str, 
    barcodes_tsv_file: str, sample: int=None, save_to_file: bool=False, 
    path_to_save: str=None) -> pd.DataFrame:
    '''Converts single cell data to reads table where columns are the cells and rows are gene reads.
        Performs sampling if sample is not None.

    Args:
        matrix_mtx_file: path to matrix.mtx.
        features_tsv_file: path to features.tsv.
        barcodes_tsv_file: path to barcodes.tsv.
        sample: column sample size, default = None.
        save_to_file: save generated data table to file at new_txt_file, default = False.
        path_to_save: path to save the generated data table if save_to_file is True, default = None.

    Returns:
        Reads table.
    '''
    logging.debug('Converting visium data to reads table')
    data = ad.read_mtx(matrix_mtx_file, dtype='int32').T

    with open(features_tsv_file) as fd:
        rd = csv.reader(fd, delimiter='\t', quotechar='"')
        genes = []
        for row in rd:
            genes.append(row[1])

    with open(barcodes_tsv_file) as fd:
        rd = csv.reader(fd, delimiter='\t', quotechar='"')
        cells = []
        for row in rd:
            cells.append(row[0])

    data.obs = pd.DataFrame(index=cells)
    data.var = pd.DataFrame(index=genes)
    if sample is not None and data.n_obs > sample:
        rng = np.random.default_rng(seed=None)
        rand = rng.integers(0,100,1)[0]
        sc.pp.subsample(data, n_obs=sample, random_state=rand)
    data = data.T
    data = data.to_df()

    if save_to_file:
        data.to_csv(path_to_save, sep='\t')

    return data


def switch_10x_to_txt(matrix_mtx_file: str, features_tsv_file: str, 
    barcodes_tsv_file: str, save_to_file: bool=False, 
    path_to_save: str=None) -> pd.DataFrame:
    '''Converts visium data to reads table where columns are the spots and rows are gene reads.

    Args:
        matrix_mtx_file: path to matrix.mtx that was downloaded from visium website.
        features_tsv_file: path to features.tsv that was downloaded from visium website.
        barcodes_tsv_file: path to barcodes.tsv that was downloaded from visium website.
        save_to_file: save generated data table to file at new_txt_file, default = False.
        path_to_save: path to save the generated data table if save_to_file is True, default = None.

    Returns:
        Reads table.
    '''
    logging.debug('Converting visium data to reads table')
    the_matrix = scipy.io.mmread(matrix_mtx_file).todense()

    with open(features_tsv_file) as fd:
        rd = csv.reader(fd, delimiter='\t', quotechar='"')
        genes = []
        for row in rd:
            genes.append(row[1])

    with open(barcodes_tsv_file) as fd:
        rd = csv.reader(fd, delimiter='\t', quotechar='"')
        cells = []
        for row in rd:
            cells.append(row[0])

    data = pd.DataFrame(the_matrix, columns=cells, index=genes)

    if save_to_file:
        data.to_csv(path_to_save, sep='\t')
    
    return data


def visium_loader(data_path: str) -> pd.DataFrame:
    '''Loads visium data.

    Args:
        data_path: path to dataset folder.
    
    Returns:
        Reads table with spots (columns) and genes (rows).
    '''
    logging.debug('Loading visium data') 
    counts_path = os.path.join(data_path, constants._SPATIAL_FOLDER_2)
    matrix_mtx_file = os.path.join(counts_path, 'matrix.mtx')
    features_tsv_file = os.path.join(counts_path, 'features.tsv')
    barcodes_tsv_file = os.path.join(counts_path, 'barcodes.tsv')
    counts = switch_10x_to_txt(matrix_mtx_file, features_tsv_file, barcodes_tsv_file)

    return counts


def load_coors_configuration_1(spatial_path: str, counts: pd.DataFrame) -> Optional[pd.DataFrame]:
    '''Loads spatial coordinates with configuration #1.

    Args:
        spatial_path: path to 'spatial' folder.
        counts: reads table.

    Returns: 
        Spatial coordinates or None.
    '''
    try:
        spatial_coors = pd.read_csv(
            os.path.join(spatial_path, 'tissue_positions_list.csv'), 
            index_col=0, 
            sep=',', 
            header='infer')
        spatial_coors = spatial_coors.loc[list(counts)]
        spatial_coors = np.array(spatial_coors[['array_col','array_row']])
        return spatial_coors
    except:
        logging.debug('Failed loading spatial coordinates with configuration #1')

def load_coors_configuration_2(spatial_path: str, counts: pd.DataFrame) -> Optional[pd.DataFrame]:
    '''Loads spatial coordinates with configuration #2.

    Args:
        spatial_path: path to 'spatial' folder.
        counts: reads table.

    Returns: 
        Spatial coordinates or None.
    '''
    try:
        spatial_coors = pd.read_csv(
            os.path.join(spatial_path, 'tissue_positions_list.csv'), 
            index_col=0, 
            sep=',',
            header=None) 
        spatial_coors = spatial_coors.loc[list(counts)]
        spatial_coors = np.array(spatial_coors[[3, 2]])
        return spatial_coors
    except:
        logging.debug('Failed loading spatial coordinates with configuration #2')

def load_coors_configuration_3(spatial_path: str, counts: pd.DataFrame) -> Optional[pd.DataFrame]:
    '''Loads spatial coordinates with configuration #3.

    Args:
        spatial_path: path to 'spatial' folder.
        counts: reads table.

    Returns: 
        Spatial coordinates or None.
    '''
    try:  
        spatial_coors = pd.read_csv(
            os.path.join(spatial_path, 'tissue_positions.csv'), 
            index_col=0, 
            sep=',', 
            header='infer')
        spatial_coors = spatial_coors.loc[list(counts)]
        spatial_coors = np.array(spatial_coors[['array_col','array_row']])
        return spatial_coors
    except:
        logging.debug('Failed loading spatial coordinates with configuration #3')

def get_spatial_coors(data_path: str, counts: pd.DataFrame) -> pd.DataFrame:
    '''Loads spatial coordinates.

    Args:
        data_path: path to dataset folder.
        counts: reads table.

    Returns:
        Spatial coordinates.

    Raises:
        TypeError when loading spatial coordinates is unsuccessful
    '''

    logging.debug('Getting spatial coordinates')

    spatial_path = os.path.join(data_path, constants._SPATIAL_FOLDER_1)
    spatial_coors = load_coors_configuration_1(spatial_path, counts)
    if spatial_coors is not None:
        return spatial_coors
    else:
        logging.debug('Retrying with configuration #2')

    spatial_coors = load_coors_configuration_2(spatial_path, counts)
    if spatial_coors is not None:
        return spatial_coors
    else:
        logging.debug('Retrying with configuration #3')

    spatial_coors = load_coors_configuration_3(spatial_path, counts)
    if spatial_coors is not None:
        return spatial_coors
    else:
        raise TypeError('Problem loading %s coordinates' %constants._DATA_TYPE_SPATIAL)    

def load_merge_txt_files(txt_files: list) -> pd.DataFrame:
    '''
    Loads and merges txt files.

    Args:
        txt_files: list of detected txt files.

    Returns:
        counts: merged data.
    '''
    counts = pd.read_csv(txt_files[0], delimiter='\t', index_col=0) 
    len_files = len(txt_files)
    if len_files > 1:
        logging.info('Merging all %i .txt files', len_files)
        for file in txt_files[1:]:
            counts_to_merge = pd.read_csv(file, delimiter='\t', index_col=0)        
            counts = counts.merge(counts_to_merge, left_index=True, right_index=True)   
    return counts     

def load_merge_tsv_files(tsv_files: list) -> pd.DataFrame:
    '''
    Loads and merges tsv files.

    Args:
        tsv_files: list of detected tsv files.

    Returns:
        counts: merged data.
    '''
    counts = pd.read_csv(tsv_files[0], sep='\t', index_col=0, on_bad_lines='skip')
    # remove after validation with other tsv's, making sure this is the right way
    # counts = pd.read_csv(tsv_files[0], sep='\t', index_col=0, on_bad_lines='skip').T 
    len_files = len(tsv_files)
    if len_files > 1:
        logging.info('Merging all %i .tsv files', len_files)
        for file in tsv_files[1:]:
            counts_to_merge = pd.read_csv(file, sep='\t', index_col=0, on_bad_lines='skip')
            # remove after validation with other tsv's, making sure this is the right way
            # counts_to_merge = pd.read_csv(file, sep='\t', index_col=0, on_bad_lines='skip').T 
            counts = counts.merge(counts_to_merge, left_index=True, right_index=True)        

    return counts     

def load_merge_10x_files(mtx_files: list, sample_size: Optional[int]=constants._MAX_COLS) -> pd.DataFrame:
    '''
    Loads and merges 10x files.

    Assuming there are three files for merge: xxx_barcodes.tsv, xxx_genes.tsv, xxx*.mtx.
    If needed, matrices are sampled during the process, so that maximum 10K columns are 
    left at the end. Each reads table is sampled with 10K/num_of_tables cells.
    If duplicated genes are found they are discarded.
    Inner merge is done if there are multiple files.

    Args:
        mtx_files: list of detected mtx files.
        sample_size: (optional) amount of cells to sample

    Returns:
        counts: merged data.
    '''
    files_prefix = '_'.join(mtx_files[0].split('/')[-1].split('_')[:-1])
    file_path = '/'.join(mtx_files[0].split('/')[:-1])
    len_files = len(mtx_files)
    matrix_mtx_file = mtx_files[0]
    relative_sampling = int(sample_size/len_files)
    features_tsv_file = os.path.join(file_path, files_prefix + _10X_SCRNASEQ_FEATURES_SUFFIX)
    barcodes_tsv_file = os.path.join(file_path, files_prefix + _10X_SCRNASEQ_BARCODES_SUFFIX)
    counts = switch_10x_to_txt_sc(
        matrix_mtx_file, 
        features_tsv_file, 
        barcodes_tsv_file, 
        sample=relative_sampling)
    counts = counts[~counts.index.duplicated(keep='first')]

    if len_files > 1:
        logging.info('Merging all %i mtx files', len_files)
        for file in mtx_files[1:]:
            files_prefix = '_'.join(file.split('/')[-1].split('_')[:-1])
            matrix_mtx_file = file
            features_tsv_file = os.path.join(
                file_path, files_prefix + _10X_SCRNASEQ_FEATURES_SUFFIX)
            barcodes_tsv_file = os.path.join(
                file_path, files_prefix + _10X_SCRNASEQ_BARCODES_SUFFIX)
            counts_to_merge = switch_10x_to_txt_sc(
                matrix_mtx_file, features_tsv_file, barcodes_tsv_file, sample=relative_sampling)
            logging.debug('%s columns were detected' %len(counts_to_merge.columns))
            counts_to_merge = counts_to_merge[~counts_to_merge.index.duplicated(keep='first')]
            counts = pd.merge(
                counts, counts_to_merge, left_index=True, right_index=True, copy=False)
    return counts     

def uzip_files(gz_files: list) -> None:
    '''If '.gz' files are found, unzip them all.

    Args:
        gz_files: files to unzip
    
    Returns:
        None
    '''

    for file in gz_files:
        output_path = file[:-3]
        with gzip.open(file,"rb") as f_in, open(output_path,"wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def scRNAseq_preprocess_loader(dataset_name: str, data_path: str, 
    sample_size: Optional[int]=constants._MAX_COLS) -> pd.DataFrame:
    '''Preprocesses and loads scRNAseq data.

    Merges all txt or tsv tables found in data_path and then samples 10K columns.
    Saves under the name given in dataset_name with .pkl extension

    Args:
        dataset_name: dataset name.
        data_path: path to dataset folder.
        sample_size: (optional) amount of cells to sample

    Returns:
        Reads table with cells (columns) and genes (rows).

    Raises:
        UsageError: if no 'txt', 'tsv', or 'mtx' files are found 
    '''
    logging.debug('Preprocessing %s data' %constants._DATA_TYPE_SINGLE_CELL)
    path_to_gz = '%s/*.gz' %data_path
    gz_files = glob.glob(path_to_gz)
    if gz_files:
        logging.info('Unzipping all files')
        uzip_files(gz_files)
    path_to_txt = '%s/*.txt' %data_path
    path_to_tsv = '%s/*.tsv' %data_path
    path_to_mtx = '%s/*.mtx' %data_path
    path_to_pkl = '%s/*.pkl' %data_path
    print(data_path)
    txt_files = glob.glob(path_to_txt)
    tsv_files = glob.glob(path_to_tsv)
    mtx_files = glob.glob(path_to_mtx)
    pkl_files = glob.glob(path_to_pkl)
    print(pkl_files)
    if pkl_files:
        counts = pd.read_pickle(pkl_files)
    elif txt_files:
        counts = load_merge_txt_files(txt_files)
    elif mtx_files:
        counts = load_merge_10x_files(mtx_files, sample_size)
    elif tsv_files:
        counts = load_merge_tsv_files(tsv_files)
    else:
        raise UsageError('No \'txt\', \'tsv\' or \'mtx\' files were found in %s. ' %data_path)
    
    len_cols = len(counts.columns)
    if len_cols > sample_size:
        logging.info('%s columns were detected' %len_cols)
        logging.info('Sampling %i columns' %sample_size)
        counts = counts.sample(n=sample_size, axis='columns')
    
    path_to_pkl = os.path.join(data_path, dataset_name + '.pkl')
    counts.to_pickle(path_to_pkl)
    
    return counts

def scRNAseq_loader(data_path: str) -> pd.DataFrame:
    '''Loads scRNAseq data.

    Args:
        data_path: path to dataset folder.

    Returns:
        Reads table with cells (columns) and genes (rows).
    '''
    logging.debug('Loading %s data' %constants._DATA_TYPE_SINGLE_CELL)

    file_name = detect_data_file(data_path)
    if file_name.endswith('.txt'):
        counts = pd.read_csv(file_name, delimiter='\t', index_col=0)        
    elif file_name.endswith('.tsv'):
        counts = pd.read_csv(file_name, sep='\t', index_col=0, on_bad_lines='skip')
        # counts = pd.read_csv(file_name, sep='\t', index_col=0, on_bad_lines='skip').T  
    else: #pkl file
        counts = pd.read_pickle(file_name)
    col_len = len(counts.columns)
    if col_len > constants._MAX_COLS:
        logging.info('Reads table is too big, having: %i columns, this might take too long '
                     'compute microRNA activity. Please consider sampling data up to %i columns, '
                     'by passing \'process\'=True' %(col_len, constants._MAX_COLS))
    return counts


def normalize_counts(counts: pd.DataFrame) -> pd.DataFrame:
    '''Normalizing reads table.

    Removing genes (rows) where all reads are zero.
    Normalizing every spot/cell (column) such that they have the same amount of total reads.
    Z-score transformation per gene (row).
    
    Args: 
        counts: reads table.

    Returns: 
        Normalized reads table.
    '''
    logging.debug('Normalizing reads table')

    counts_norm = counts.loc[counts.sum(axis=1) > 0]
    counts_norm = counts_norm.divide(
        counts_norm.sum(), axis='columns').multiply(10000)
    counts_norm = counts_norm.fillna(0)
    counts_norm = counts_norm.subtract(
        counts_norm.mean(axis=1), axis='index').divide(
            counts_norm.std(axis=1), axis='index')
    return counts_norm


def compute_mir_activity(counts: pd.DataFrame, miR_list: list, 
    mti_data: pd.DataFrame, results_path: str, cpus: int, 
    debug: bool=False) -> pd.DataFrame:
    '''Computing microRNA activity.

    Multiprocessing of per cell/spot per microRNA computations.
    Using mHG test.

    Args:
        counts: normalized reads table.
        miR_list: microRNAs to consider.
        mti_data: microRNA targets data.
        results_path: path to save results.
        cpus: amount of cpus to use in parallel.

    Returns: 
        MicroRNA activity results achived using mHG test.
    '''
    logging.debug('Initializing results tables')
    miR_activity_stats = pd.DataFrame(columns=list(counts), index=miR_list)
    miR_activity_pvals = pd.DataFrame(columns=list(counts), index=miR_list)
    miR_activity_cutoffs = pd.DataFrame(columns=list(counts), index=miR_list)

    path_to_stats = ('%s/activity_stats.csv' %results_path)
    path_to_pvals = ('%s/activity_pvals.csv' %results_path)
    path_to_cutoffs = ('%s/activity_cutoffs.csv' %results_path)

    logging.debug('Computing activity map')

    with MyPool(cpus) as pool:
        iterable = [(
            cell, counts.loc[:, cell].sort_values(), 
            miR_list, mti_data, debug) for cell in list(counts)]
        result_list = list(tqdm.tqdm(
            pool.istarmap(compute_stats_per_cell, iterable), total=len(iterable)))

    for result in result_list:
        cell = result[0]
        miR_activity_stats.loc[:, cell] = result[1]
        miR_activity_pvals.loc[:, cell] = result[2]
        miR_activity_cutoffs.loc[:, cell] = result[3]

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    miR_activity_stats.to_csv(path_to_stats)
    miR_activity_pvals.to_csv(path_to_pvals)
    miR_activity_cutoffs.to_csv(path_to_cutoffs)

    return miR_activity_pvals

def compute_stats_per_cell(cell: str, ranked: pd.DataFrame, miR_list: list, mti_data: pd.DataFrame, 
    debug: bool=False) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Computing microRNA activity per spot/cell.

    Using mHG test.
    If no targets are found in the reads table for a particular microRNA, 
    the result will be p-value of 0 for all cells/spots.
    If there are too many genes for the mHG test to handle, the list is cut to the maximum allowed, 
    removing the highly expressed genes from the bottom of the list.

    Args:
        cell:  column id.
        ranked: sorted reads column for the cell.
        miR_list: microRNAs to consider.
        mti_data: microRNA targets data.

    Returns: 
        Column id and microRNA activity results achived using mHG test.
    '''
    logging.debug('Computing statistics for %s ' %cell)

    ranked_list = list(ranked.index)
    miR_activity_stats = []
    miR_activity_pvals = []
    miR_activity_cutoffs = []
    len_ranked = len(ranked)
    for miR in miR_list:
        miR_targets = list(mti_data[mti_data['miRNA'] == miR]['Target Gene'])
        v = np.uint8([int(g in miR_targets) for g in ranked_list])
        if debug:
            if sum(v) == 0: 
                logging.debug('No targets found for %s' %miR)
                logging.debug('No targets were found for microRNA: %s . please check that '
                                 'correct \'species\' flag was selected. The following '
                                 'targets were found: %s. ' %(miR, miR_targets))
        stat, cutoff, pval = xlmhg.xlmhg_test(v, X=_MHG_X_PARAM, L=len_ranked)
        miR_activity_stats.append(stat)
        miR_activity_cutoffs.append(cutoff)
        miR_activity_pvals.append(pval)

    return cell, miR_activity_stats, miR_activity_pvals, miR_activity_cutoffs

def get_figure_list(miR_list: list, miR_figures: str, mir_activity_list: pd.DataFrame) -> list:
    '''Checking which microRNAs to plot.

    Args:
        miR_list: list of microRNAs.
        miR_figures: which microRNAs the user would like to plot, default: top 10 most active. 
        mir_activity_list: Sorted list of microRNAs according to their overall activity.

    Return:
        List of microRNA's to plot.
    '''
    if miR_figures == constants._DRAW_ALL or len(miR_list) <= 10:
        miR_list_figures = miR_list
    elif miR_figures == constants._DRAW_TOP_10:
        miR_list_figures = mir_activity_list.index[:10].tolist()
    else: #'bottom_10'
        miR_list_figures = mir_activity_list.index[-10:].tolist()
    logging.debug('Figures are produced for the following microRNAs: %s', miR_list_figures)

    return miR_list_figures

def sort_activity_spatial(miR_activity_pvals: pd.DataFrame, thresh: float, 
    spots: int, results_path: str, dataset_name: str) -> pd.DataFrame:
    '''Computes which microRNAs are the most active within the entire slide.

    The value of each microRNA is the percentage of cells/spots 
    for which it got a score lower than 'thresh'

    Args:
        miR_activity_pvals: activity table per spot per microRNA.
        thresh: used for filtering only microRNAs/spt that got a score 
            lower than thresh, i.e. very active.
        spots: number of spots.
        results_path: where to save the sorted list of most highly expressed microRNA.
        dataset_name: dataset name
    
    Returns: 
        Sorted list of microRNA, from the most overall active to the least, over the extire slide. 
    '''
    logging.info('Computing which microRNAs are the most active within the entire slide.')

    mir_activity_list = miR_activity_pvals[
        miR_activity_pvals < thresh].count(axis=1).sort_values(ascending=False)
    mir_activity_list = mir_activity_list / spots
    mir_activity_list = pd.DataFrame(mir_activity_list)
    mir_activity_list.columns = ['Activity Score']
    mir_activity_list = mir_activity_list.rename_axis('MicroRNA')

    return mir_activity_list

def produce_spatial_maps(miR_list_figures: list, miR_activity_pvals: pd.DataFrame, 
    spatial_coors: pd.DataFrame, results_path: str, dataset_name: str, 
    mir_activity_list: pd.DataFrame):
    '''Produces a figure with activity map per microRNA in the list.
       Produces also a html file with list of microRNAs, sorted by their overall activity.

    Args:
        miR_list_figures: list of microRNAs to produce figures for.
        miR_activity_pvals: activity per spot per microRNA.
        spatial_coors: spatial location of each spot.
        results_path: path to save figures.
        dataset_name: for plot name.
        mir_activity_list: sorted list of most active microRNAs.
    
    Returns:
        None
    '''
    logging.debug('Generating figures')
    results_path_figures = os.path.join(results_path, 'activity maps')
    plot_label = 'p-value (-log10)'
    path_to_list = '%s/sorted_mir_by_activity.html' %results_path

    if not os.path.exists(results_path_figures):
        os.makedirs(results_path_figures)

    for miR in miR_list_figures:
        plot_title = '%s activity map' %miR
        pvals = miR_activity_pvals.loc[miR, :]
        log10_pvals = -np.log10(pvals)
        plot_file_name = '%s_%s.jpg' %(dataset_name, miR)
        path_to_plot = '%s/%s' %(results_path_figures, plot_file_name)
        plt.figure(figsize=(10, 10))
        plt.scatter(spatial_coors[:, 0], spatial_coors[:, 1], c=log10_pvals, 
            vmin=np.min(log10_pvals), vmax=np.max(log10_pvals))
        plt.gca().invert_yaxis()
        plt.colorbar(extend='max').set_label(plot_label, rotation=270, labelpad=20)
        plt.title(plot_title, fontsize=14)
        plt.savefig(path_to_plot)
        logging.debug('Figure generated for %s, saved in %s' %(miR, path_to_plot))
        ref_path = '"./activity maps/%s"' %plot_file_name 
        index_rename = '<a href=%s target="_blank">%s</a>' %(ref_path, miR)
        mir_activity_list = mir_activity_list.rename(index={miR:index_rename})
    mir_activity_list = mir_activity_list.reset_index()
    mir_activity_list.to_html(path_to_list, escape=False, index=False, justify='left')

def generate_umap(counts: pd.DataFrame, miR_activity_pvals: pd.DataFrame, 
    populations: Optional[list]=None) -> sc.AnnData:
    '''Computing UMAP based on gene counts, and enriching it with miR activity scores.

    Args:
        counts: reads table.
        miR_activity_pvals: activity per spot per microRNA.
        populations: (optional) list of two population string identifiers embedded in cell id.

    Returns:
        Enriched counts table with umap and miR activity scores.
    '''
    logging.info('Normalizing gene expression.')
    enriched_counts = sc.AnnData(counts.T)
    if populations:
        categories = pd.DataFrame(counts.columns,columns=['cell_id'])
        enriched_counts.obs['populations'] = pd.Categorical(
            np.where(categories['cell_id'].str.contains(
                populations[0]), populations[0], populations[1]))
    sc.pp.filter_cells(enriched_counts, min_genes=200)
    sc.pp.filter_genes(enriched_counts,min_cells=3)
    sc.pp.calculate_qc_metrics(enriched_counts, percent_top=None, log1p=False, inplace=True)
    upper_lim = np.quantile(enriched_counts.obs.n_genes_by_counts.values, .98)
    lower_lim = np.quantile(enriched_counts.obs.n_genes_by_counts.values, .02)
    enriched_counts = enriched_counts[
        (enriched_counts.obs.n_genes_by_counts<upper_lim)&
        (enriched_counts.obs.n_genes_by_counts>lower_lim)]
    sc.pp.normalize_total(enriched_counts, target_sum=1e4)
    sc.pp.log1p(enriched_counts)
    sc.pp.highly_variable_genes(enriched_counts, min_mean=0.0125, max_mean=3, min_disp=0.5)
    enriched_counts.raw=enriched_counts
    enriched_counts = enriched_counts[:, enriched_counts.var.highly_variable]
    sc.pp.regress_out(enriched_counts, ['total_counts'])
    sc.pp.scale(enriched_counts, max_value=10)
    logging.info('Computing UMAP.')
    sc.tl.pca(enriched_counts, svd_solver='arpack') 
    sc.pp.neighbors(enriched_counts, n_neighbors=10, n_pcs=30)
    sc.tl.umap(enriched_counts)
    logging.info('Enriching with microRNA activity results.')
    log10_pvals = -np.log10(miR_activity_pvals)
    mir_for_merge = sc.AnnData(log10_pvals.T)
    adatas = []
    adatas.append(enriched_counts)
    adatas.append(mir_for_merge)
    adata_merge = ad.concat(adatas, axis=1, merge="unique")
    return adata_merge

def sort_activity_sc(miR_activity_pvals: pd.DataFrame, 
    populations: Optional[list]=None) -> pd.DataFrame:
    '''Calls a sorting method according to availability of populations.

    Args:
        miR_activity_pvals: activity table per spot per microRNA.
        populations: (optional) list of two population string identifiers embedded in cell id.

    Returns:
        Sorted list of microRNA, from the most overall active to the least, over all cells. 
    '''
    logging.info('Sorting microRNAs accoridng to their activity patterns.')
    if populations:
        mir_activity_list = sort_activity_sc_with_populations(miR_activity_pvals, populations)
    else:
        mir_activity_list = sort_activity_sc_no_populations(miR_activity_pvals)
    return mir_activity_list

def sort_activity_sc_no_populations(miR_activity_pvals: pd.DataFrame) -> pd.DataFrame:
    '''Sorts microRNA's by their overall activity.

    Args:
        miR_activity_pvals: activity table per spot per microRNA.

    Returns:
        Sorted list of microRNA, from the most overall active to the least, over all cells. 
    '''
    log10_pvals = -np.log10(miR_activity_pvals)
    mir_activity_list = log10_pvals[
        log10_pvals > _NON_ACTIVE_THRESH].mean(axis=1).sort_values(ascending=False)
    mir_activity_list = pd.DataFrame(mir_activity_list)
    mir_activity_list.columns = ['Activity Score']
    mir_activity_list = mir_activity_list.rename_axis('MicroRNA')
    return mir_activity_list

def sort_activity_sc_with_populations(miR_activity_pvals: pd.DataFrame, 
    populations: list) -> pd.DataFrame:
    '''Sorts active microRNAs according to their differential expression between populations.

    Args:
        miR_activity_pvals: activity table per spot per microRNA.
        populations: list of two population string identifiers embedded in cell id.

    Returns:
        Sorted list of active microRNA, according to their differential expression 
        between populations. 
    '''
    log10_pvals = -np.log10(miR_activity_pvals)
    pop_1_cols = [col for col in log10_pvals.columns if (populations[0] in col)]
    pop_2_cols = [col for col in log10_pvals.columns if (col not in pop_1_cols)]

    # pop_2_cols = [col for col in log10_pvals.columns if (populations[1] in col)]
    col_name_pop_1 = 'mean_' + populations[0]
    col_name_pop_2 = 'mean_' + populations[1]
    miR_list = miR_activity_pvals.index.tolist()
    mir_amount = len(miR_list)
    mir_activity_list = pd.DataFrame(
        columns=['ranksum_pval', col_name_pop_1, col_name_pop_2, 'fdr_corrected'], 
        index=miR_list).astype('float64')
    for miR in miR_list:
        pvals_pop_1 =  log10_pvals.loc[miR, pop_1_cols]
        pvals_pop_2 =  log10_pvals.loc[miR, pop_2_cols]
        stat, pval = ranksums(
            pvals_pop_1[pvals_pop_1 > _NON_ACTIVE_THRESH], 
            pvals_pop_2[pvals_pop_2 > _NON_ACTIVE_THRESH])
        if np.isnan(pval):
            mir_activity_list['ranksum_pval'][miR] = 1
            mir_activity_list[col_name_pop_1][miR] = pvals_pop_1.mean()
            mir_activity_list[col_name_pop_2][miR] = pvals_pop_2.mean()
        else:
            mir_activity_list['ranksum_pval'][miR] = pval
            mir_activity_list[col_name_pop_1][miR] = \
                pvals_pop_1[pvals_pop_1 > _NON_ACTIVE_THRESH].mean()
            mir_activity_list[col_name_pop_2][miR] = \
                pvals_pop_2[pvals_pop_2 > _NON_ACTIVE_THRESH].mean()

    mir_activity_list = mir_activity_list.sort_values(by=['ranksum_pval'])
    for i in range(len(mir_activity_list)):
        if mir_activity_list.iloc[i]['ranksum_pval'] < 1:
            mir_activity_list.iloc[i]['fdr_corrected'] = \
                mir_activity_list.iloc[i]['ranksum_pval']*mir_amount/(i+1)

    mean_all = mir_activity_list[col_name_pop_1].append(mir_activity_list[col_name_pop_2])
    very_active = mean_all.quantile(0.97)
    active = mean_all.quantile(0.9)

    pval_mask_active = (
        (mir_activity_list['fdr_corrected'] < 1e-8) & 
        ((mir_activity_list[col_name_pop_1] >= active) | 
        (mir_activity_list[col_name_pop_2] >= active)))

    pval_mask_very_active = (
        (mir_activity_list['fdr_corrected'] < 1e-8) & 
        ((mir_activity_list[col_name_pop_1] >= very_active) | 
        (mir_activity_list[col_name_pop_2] >= very_active)))
    
    mir_activity_list_high_pval = mir_activity_list[pval_mask_very_active]
    mir_activity_list_mid_pval = mir_activity_list[pval_mask_active & ~pval_mask_very_active]
    mir_activity_list_low_pval = mir_activity_list[~pval_mask_active]
    mir_activity_list = pd.concat([mir_activity_list_high_pval, mir_activity_list_mid_pval, mir_activity_list_low_pval])
    mir_activity_list = mir_activity_list.rename_axis('MicroRNA')
    return mir_activity_list

def plot_sc(miR_list_figures: list, enriched_counts: sc.AnnData, results_path: str, 
    dataset_name: str, mir_activity_list: pd.DataFrame, miR_activity_pvals: pd.DataFrame, 
    populations: Optional[list]=None):
    '''Calls plotting functions according to availability of populations data.

    Args:
        miR_list_figures: list of microRNAs to produce figures for.
        enriched_counts: enriched counts table with umap and miR activity scores.
        results_path: path to save figures.
        dataset_name: for plot name.
        mir_activity_list: sorted list of most active microRNAs.
        miR_activity_pvals: activity table per cell per microRNA.
        populations: (optional) list of two population string identifiers embedded in cell id.

    Returns:
        None
    '''
    if populations:
        plot_sc_with_populations(        
            miR_list_figures, 
            enriched_counts,
            results_path, 
            dataset_name, 
            mir_activity_list,
            miR_activity_pvals,
            populations)
    else:
        plot_sc_no_populations(
            miR_list_figures, 
            enriched_counts,
            results_path, 
            dataset_name, 
            mir_activity_list)

def plot_sc_no_populations(miR_list_figures: list, enriched_counts: sc.AnnData, results_path: str, 
    dataset_name: str, mir_activity_list: pd.DataFrame):
    '''Produces UMAP figure with microRNA activity levels per microRNA in the list.
       Produces also a html file with list of microRNAs, sorted by their overall activity.

    Args:
        miR_list_figures: list of microRNAs to produce figures for.
        enriched_counts: enriched counts table with umap and miR activity scores.
        results_path: path to save figures.
        dataset_name: for plot name.
        mir_activity_list: sorted list of most active microRNAs.
    
    Returns:
        None
    '''
    logging.debug('Generating UMAP figures')
    results_path_figures = os.path.join(results_path, 'activity maps')
    path_to_list = '%s/sorted_mir_by_activity.html' %results_path

    if not os.path.exists(results_path_figures):
        os.makedirs(results_path_figures)

    for miR in miR_list_figures:
        plot_file_name = '%s_%s_%s.jpg' %(dataset_name, miR, 'umap')
        path_to_plot = '%s/%s' %(results_path_figures, plot_file_name)
        fig, ax = plt.subplots(figsize=(10, 10))
        sc.pl.umap(enriched_counts, color=miR, title='%s activity (-log10)' %miR, show=False, ax=ax)
        fig.savefig(path_to_plot)
        logging.debug('Figure generated for %s, saved in %s' %(miR, path_to_plot))
        ref_path = '"./activity maps/%s"' %plot_file_name 
        index_rename = '<a href=%s target="_blank">%s</a>' %(ref_path, miR)
        mir_activity_list = mir_activity_list.rename(index={miR:index_rename})
    mir_activity_list = mir_activity_list.reset_index()
    mir_activity_list.to_html(path_to_list, escape=False, index=False, justify='left')

def plot_sc_with_populations(miR_list_figures: list, enriched_counts: sc.AnnData, results_path: str, 
    dataset_name: str, mir_activity_list: pd.DataFrame, miR_activity_pvals: pd.DataFrame, 
    populations: list):
    '''Produces UMAP figure and histogram with microRNA activity levels per microRNA in the list.
       Produces also a html file with list of active microRNAs, sorted by their 
       differential expression between the populations.

    Args:
        miR_list_figures: list of microRNAs to produce figures for.
        enriched_counts: enriched counts table with umap and miR activity scores.
        results_path: path to save figures.
        dataset_name: for plot name.
        mir_activity_list: sorted list of most active microRNAs.
        miR_activity_pvals: activity table per cell per microRNA.
        populations: list of two population string identifiers embedded in cell id.

    Returns:
        None
    '''
    logging.debug('Generating figures')
    results_path_figures = os.path.join(results_path, 'activity maps')
    path_to_list = '%s/sorted_mir_by_activity.html' %results_path

    if not os.path.exists(results_path_figures):
        os.makedirs(results_path_figures)

    for miR in miR_list_figures:
        umap_plot_file_name = '%s_%s_%s_%s_%s.jpg' %(
            dataset_name, miR, populations[0], populations[1], 'umap')
        path_to_umap_plot = '%s/%s' %(results_path_figures, umap_plot_file_name)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sc.pl.umap(enriched_counts, size=10, show=False, ax=ax1)
        sc.pl.umap(
            enriched_counts[enriched_counts.obs['populations']==populations[0]], 
            size=100,
            color=miR, 
            title='%s %s activity (-log10)' %(populations[0], miR), 
            show=False, 
            ax=ax1)
        sc.pl.umap(enriched_counts, size=10, show=False, ax=ax2)
        sc.pl.umap(
            enriched_counts[enriched_counts.obs['populations']==populations[1]], 
            size=100,
            color=miR, 
            title='%s %s activity (-log10)' %(populations[1], miR), 
            show=False, 
            ax=ax2)
        fig.savefig(path_to_umap_plot)
        logging.debug('Figure generated for %s, saved in %s' %(miR, path_to_umap_plot))
        ref_umap_path = '"./activity maps/%s"' %umap_plot_file_name 
        index_rename = '<a href=%s target="_blank">%s</a>' %(ref_umap_path, miR)
        
        kwargs = dict(alpha=0.5, bins=100, density=True, stacked=False, histtype="bar")
        legend_loc = 'upper right'
        hist_plot_file_name = '%s_%s_%s_%s_%s.jpg' %(
            dataset_name, miR, populations[0], populations[1], 'histogram')
        path_to_hist_plot = '%s/%s' %(results_path_figures, hist_plot_file_name )
        log10_pvals = -np.log10(miR_activity_pvals)
        pop_1_cols = [col for col in log10_pvals.columns if (populations[0] in col)]
        pop_2_cols = [col for col in log10_pvals.columns if (populations[1] in col)]
        pvals_pop_1 =  log10_pvals.loc[miR, pop_1_cols]
        pvals_pop_2 =  log10_pvals.loc[miR, pop_2_cols]
        wrk_fdr_result = mir_activity_list.loc[miR]['fdr_corrected']
        f, ax = plt.subplots()
        plt.figure(figsize = (20, 10))
        f.subplots_adjust(top=0.7)
        plt.hist((pvals_pop_1, pvals_pop_2), **kwargs, label = (populations[0], populations[1]))
        plt.xlabel('p-value (-log10)', fontsize = 26)
        plt.ylabel('Probability', fontsize = 26)
        plt.legend(loc=legend_loc, prop={'size': 28})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('%s Activity Histogram \n%s Vs. %s' %(miR, populations[0], populations[1]), 
                fontsize=34, pad=40, y=0.95)
        f.tight_layout(pad = 0.5)     
        plt.savefig(path_to_hist_plot)
        plt.close()
        logging.debug('Figure generated for %s, saved in %s' %(miR, path_to_hist_plot))
        ref_hist_path = '"./activity maps/%s"' %hist_plot_file_name 
        score_col_rename = '<a href=%s target="_blank">%s</a>' %(ref_hist_path, wrk_fdr_result)
        mir_activity_list['fdr_corrected'][miR] = score_col_rename
        mir_activity_list = mir_activity_list.rename(index={miR:index_rename})

    for miR in mir_activity_list.iloc[10:].index.to_list():
        mir_activity_list['fdr_corrected'][miR] = str(mir_activity_list['fdr_corrected'][miR])
    mir_activity_list = mir_activity_list.reset_index()
    mir_activity_list_to_user = mir_activity_list[['MicroRNA', 'fdr_corrected']].copy()
    col_rename = '%s Vs. %s FDR Corrected ' %(populations[0],populations[1])
    mir_activity_list_to_user = mir_activity_list_to_user.rename(columns={'fdr_corrected': col_rename})
    mir_activity_list_to_user.to_html(path_to_list, escape=False, index=False, justify='left')