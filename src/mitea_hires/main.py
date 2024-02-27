import multiprocessing as mp
import os
from pathlib import Path
from time import time
from typing import Optional, Sequence, Tuple
import warnings

from absl import app
from absl import flags
from absl import logging
import pandas as pd

from . import constants
from . import utils

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'cpus', None, 
    ('(Optional) Number of CPUs to use in parallel. Default: all available cpus'
     ' are used.'))
flags.DEFINE_string(
    'data_path', None, 
    ('(Required) Path to your dataset. If %s: path to dataset, if %s: path to '
     'the folder containing %s folders.' % (
         constants._DATA_TYPE_SINGLE_CELL, constants._DATA_TYPE_SPATIAL, 
         ' and '.join(constants._SPATIAL_FOLDERS))))
flags.DEFINE_string(
    'dataset_name', None, 
    '(Required) Name of your dataset. This is used to generate results path.')
flags.DEFINE_boolean(
    'debug', False, '(Optional) Produces debugging output.')
flags.DEFINE_integer(
    'filter_spots', None, 
    ('(Optional) Filter spots containins total amount of reads below the '
     'specified number. Relevant only for %s data. Default: do not filter any '
     'spot.' % constants._DATA_TYPE_SPATIAL))
flags.DEFINE_string(
    'miR_figures', constants._DRAW_TOP_10, 
    ('(Optional) Which microRNA activity maps to plot from a sorted list of '
     'potentially interesting microRNAs. Options: %s .' % ' or '.join(
        constants._SUPPORTED_DRAW)))
flags.DEFINE_list(
    'miR_list', None, 
    ('(Optional) Comma-separated list of microRNAs to compute. Default: all '
     'microRNAs are considered. '
     'Example use: -miR_list=hsa-miR-300,hsa-miR-6502-5p,hsa-miR-6727-3p'))
flags.DEFINE_list(
    'populations', None,
    ('(Optional) Comma-separated list of two unique population string '
     'identifiers embedded in cell id. Required in order to compute in '
     '\'Comparative\' mode. Default: Computing in \'Total\' mode, ignoring '
     'populations. Example use: -populations=\'DISEASE\',\'CONTROL\''))
flags.DEFINE_boolean(
    'preprocess', True, 
    ('(Optional) Performs additional preprocessing on %s data before '
     'computations: merges all files found under \'data_path\' and samples %s '
     'columns. ' % (constants._DATA_TYPE_SINGLE_CELL, constants._MAX_COLS)))
flags.DEFINE_string(
    'results_path', None, 
    '(Optional) Path to save results. Default: \'data_path\'')
flags.DEFINE_integer(
    'sample_size', constants._MAX_COLS, 
    '(Optional) Amount of cells to sample in total, if \'preprocess\' is true.')
flags.DEFINE_string(
    'species', constants._SPECIES_HOMO_SAPIENS, 
    ('(Optional) %s.' %' or '.join(constants._SUPPORTED_SPECIES)))
flags.DEFINE_float(
    'thresh', constants._ACTIVITY_THRESH, 
    ('(Optional) Threshold for the microRNA activity p-value. If a microRNA at '
     'a specific spot gets a lower score than \'thresh\', it is considered '
     'active in that spot. Used in order to find the most active microRNAs '
     'across all the spots. Relevant only for %s data.' 
    % constants._DATA_TYPE_SPATIAL))

flags.register_validator('species', 
                         lambda value: value in constants._SUPPORTED_SPECIES,
                         message=('Either %s are supported.' 
                                  % (' or '.join(constants._SUPPORTED_SPECIES))))
flags.register_validator('miR_figures', 
                         lambda value: value in constants._SUPPORTED_DRAW,
                         message=('Either %s are supported.'
                                  % (' or '.join(constants._SUPPORTED_DRAW))))

flags.mark_flag_as_required('dataset_name')
flags.mark_flag_as_required('data_path')

def process_data(
        data_path: str, 
        dataset_name: str, 
        data_type: Optional[str], 
        preprocess: Optional[bool] = True, 
        filter_spots: Optional[int] = None, 
        sample_size: Optional[int] = constants._MAX_COLS
        ) -> pd.DataFrame:
    """Loading, preprocessing (if needed) and normalizing data.

    Args:
        data_path: Path to data.
        dataset_name: Dataset name.
        data_type: (optional) Data type 'spatial' or 'scRNAseq'.
        preprocess: (optional) If True (default), performing additional data 
            preprocessing (i.e. merging multiple files, and sampling if data is 
            too big). If False, data will not be preprocessed. 
        filter_spots: (optional) Filter spots containing total number of reads 
            below this number. Relevant only for spatial data. 
            Default: no filtering.
        sample_size: (optional) Amount of cells to sample, if 'preprocess' is 
            true. Relevant only for single-cell data.

    Returns:
        Normalized reads table.
        Raw reads table.
    """
    if not data_type:
       data_type =  utils.check_data_type(data_path)

    if data_type == constants._DATA_TYPE_SPATIAL:
        counts = utils.load_visium(data_path, filter_spots=filter_spots)
    else:
        if preprocess:
            counts = utils.load_single_cell_with_preprocessing(
                dataset_name, data_path, sample_size=sample_size)
        else:
            counts = utils.load_single_cell_without_preprocessing(data_path)
    counts_norm = utils.normalize_counts(counts)
    return counts_norm, counts

def compute_mir_activity(
        counts_norm: pd.DataFrame, 
        results_path: str, 
        miR_list: Optional[list] = None, 
        cpus: Optional[int] = None, 
        species: Optional[str] = constants._SPECIES_HOMO_SAPIENS, 
        debug: Optional[bool] = False
        ) -> Tuple[list, pd.DataFrame]:
    """Computing microRNA activity levels across all cells/spots.

    Args:
        counts_norm: Normalized reads table.
        results_path: Path to save results.
        miR_list: (optional) List of miRs to compute. Default: all miRs.
        cpus: (optional) Amount of cpus to use in parallel. Default: all cpus.
        species: (optional) 'homo_sapiens' (default) or 'mus_musculus'. 
        debug: (optional) If True, provides aditional information. 
            Default: False.
    
    Returns:
        List of computed microRNAs.
        MicroRNA activity score per cell/spot.
    """
    cpus = cpus or mp.cpu_count()
    logging.info('Using %i cpus.' % cpus)

    mti_data, miR_list = utils.load_mir_data(
        miR_list=miR_list, 
        species=species)
    
    start = time() 
    miR_activity_pvals = utils.compute_mir_activity(
        counts_norm, 
        miR_list, 
        mti_data, 
        results_path, 
        cpus, 
        debug=debug)
    logging.info('Computation time: %f minutes.', (time() - start)//60)
    return miR_list, miR_activity_pvals

def post_processing_spatial(
        data_path: str, 
        counts_norm: pd.DataFrame, 
        miR_activity_pvals: pd.DataFrame, 
        miR_list: list, 
        results_path: str, 
        dataset_name: str, 
        data_type: Optional[str], 
        miR_figures: Optional[str] = constants._DRAW_TOP_10, 
        thresh: Optional[float] = constants._ACTIVITY_THRESH
        ) -> None:
    """Post processing for spatial data.

    Sorting microRNAs by their abundance of activity across spots and producing 
    activity maps. 

    Args:
        data_path: Path to data.
        counts_norm: Normalized reads table.
        miR_activity_pvals: MicroRNA activity results per spot.
        miR_list: List of computed microRNAs.
        results_path: Path to save results.
        dataset_name: Dataset name.         
        data_type: (optional) Data type 'spatial' or 'scRNAseq'.
        miR_figures: (optional) Which microRNAs to plot from a sorted list of 
            potentially interesting miRs. Default: top_10.
        thresh: (optional) Thresold to define what is considered active.

    Returns:
        None.

    Raises:
        UsageError if spatial data was not found in data_path.
    """
    if not data_type:
       data_type =  utils.check_data_type(data_path)
    if data_type is not constants._DATA_TYPE_SPATIAL:
        raise utils.UsageError('No spatial data was found for post-processing '
                               'in %s.' % data_path)
    logging.info('Generating activity map figures.')
    spatial_coors = utils.get_spatial_coors(data_path, counts_norm)
    spots = spatial_coors.shape[0]

    mir_activity_list = utils.sort_activity_spatial(
        miR_activity_pvals, 
        thresh, 
        spots, 
        results_path, 
        dataset_name)

    miR_list_figures = utils.get_figure_list(
        miR_list,
        miR_figures,
        mir_activity_list)

    utils.plot_spatial_maps(
        miR_list_figures, 
        miR_activity_pvals, 
        spatial_coors, 
        results_path, 
        dataset_name, 
        mir_activity_list)

def post_processing_single_cell(
        data_path: str, 
        counts: pd.DataFrame, 
        miR_activity_pvals: pd.DataFrame, 
        miR_list: list, 
        results_path: str, 
        dataset_name: str, 
        data_type: Optional[str],
        miR_figures: Optional[str] = constants._DRAW_TOP_10, 
        populations: Optional[list] = None
        ) -> None:
    """Post processing for scRNAseq data.

    Computing UMAP based on gene expression. 
    Determined by 'Total' (populations = None) or 'Comparative' (populations 
    provided) modes, microRNAs are sorted by their potential interest, 
    and activity maps are plotted.

    Args:
        data_path: Path to data.
        counts: Raw reads table.
        miR_activity_pvals: MicroRNA activity results per cell.
        miR_list: List of computed microRNAs.
        results_path: Path to save results.
        dataset_name: Dataset name.         
        data_type: (optional) Data type 'spatial' or 'scRNAseq'.
        miR_figures: (optional) Which microRNAs to plot from a sorted list of 
            potentially interesting miRs. Default: top_10.
        populations: (optional) List of two unique population string identifiers 
            embedded in cell id. Required in order to compute in 'Comparative' 
            mode. Default: Computing in 'Total' mode,ignoring populations. 
            Example use: populations='DISEASE','CONTROL'

    Returns:
        None.

    Raises:
        UsageError if single-cell data was not found in data_path.
    """
    if not data_type:
       data_type =  utils.check_data_type(data_path)
    if data_type is not constants._DATA_TYPE_SINGLE_CELL:
        raise utils.UsageError('No scRNAseq data was found for post-processing '
                               'in %s.' % data_path)
    logging.info('Single cell post processing.')
    if not populations:
        logging.info('Computing in \'Total\' activity mode.')
    else:
        logging.info('Populations identified, computing in \'Comparative\' '
                     'activity mode.')

    enriched_counts = utils.generate_umap(
        counts, 
        miR_activity_pvals,
        populations=populations)

    mir_activity_list = utils.sort_activity_single_cell(
        miR_activity_pvals, 
        populations=populations)

    miR_list_figures = utils.get_figure_list(
        miR_list,
        miR_figures,
        mir_activity_list)

    utils.plot_single_cell(
        miR_list_figures, 
        enriched_counts,
        results_path, 
        dataset_name, 
        mir_activity_list,
        miR_activity_pvals,
        populations=populations)

def compute(
        data_path: str, 
        dataset_name: str, 
        miR_list: Optional[list] = None, 
        cpus: Optional[int] = None, 
        results_path: Optional[str] = None, 
        sample_size: Optional[int] = constants._MAX_COLS, 
        species: Optional[str] = constants._SPECIES_HOMO_SAPIENS,  
        miR_figures: Optional[str] = constants._DRAW_TOP_10, 
        preprocess: Optional[bool] = True, 
        thresh: Optional[float] = constants._ACTIVITY_THRESH,
        populations: Optional[list] = None, 
        filter_spots: Optional[int] = None,
        debug: Optional[bool] = False
        ) -> None:
    """End-to-end microRNA activity map computation.

    Loading spatial/scRNAseq data and preprocessing if needed.
    Computing microRNA activity for all spots/cells.
    Producing activity maps and saving results locally.

    Args:
        data_path: Path to data.
        dataset_name: Dataset name. 
        miR_list: (optional) List of microRNAs to compute. Default: all 
            available miRs.
        cpus: (optional) Amount of cpus to use in parallel. Default: all 
            available cpus.
        results_path: (optional) Path to save results. Default: data_path.
        sample_size: (optional) Amount of cells to sample in total 
            (default: 10K), if 'preprocess' is true. 
        species: (optional) Either 'homo_sapiens' (default) or 'mus_musculus' 
            are supported. 
        miR_figures: (optional) Which microRNAs to plot from a sorted list of 
            potentially interesting miRs. Default: top_10.
        preprocess: (optional) If True (default), performing additional data 
            preprocessing on single cell data before computations: merging all 
            files found under 'data_path' and sampling according to 
            'sample_size' if data is too big. If False, does not perform data 
            preprocessing. 
        thresh: (optional) Thresold to define what is considered an active spot 
            (only spatial data).
        populations: (optional) List of two unique population string identifiers 
            embedded in cell id. Required in order to compute in 'Comparative' 
            mode. Default: Computing in 'Total' mode, ignoring populations. 
            Example use: populations='DISEASE','CONTROL'
        filter_spots: (optional) Filter spots containing total reads below this 
            number (only spatial data).
        debug: (optional) If True, provides aditional information. 
            Default: False.

    Returns:
        None
    """
    if debug:
        logging.set_verbosity(logging.DEBUG)
        logging.debug('Debug mode is on.')
    else:
        logging.set_verbosity(logging.INFO)

    logging.info('Dataset name: %s.' % dataset_name)
    logging.info('Path to dataset: %s.' % data_path)

    if results_path:
        results_path = os.path.join(results_path, dataset_name)
    else:
        results_path = os.path.join(data_path, 'results')
    logging.info('Results path: %s.' % results_path)
    
    data_type = utils.check_data_type(data_path)
   
    counts_norm, counts = process_data(
        data_path, 
        dataset_name, 
        data_type=data_type, 
        preprocess=preprocess,
        filter_spots=filter_spots,
        sample_size=sample_size)

    miR_list, miR_activity_pvals = compute_mir_activity(
        counts_norm, 
        results_path, 
        miR_list=miR_list, 
        cpus=cpus, 
        species=species,
        debug=debug)

    if data_type == constants._DATA_TYPE_SPATIAL:
        post_processing_spatial(
            data_path, 
            counts_norm, 
            miR_activity_pvals, 
            miR_list, 
            results_path, 
            dataset_name,
            data_type=data_type,
            miR_figures=miR_figures, 
            thresh=thresh
            )
    else:
        post_processing_single_cell(
            data_path, 
            counts, 
            miR_activity_pvals, 
            miR_list, 
            results_path, 
            dataset_name,
            data_type=data_type,
            miR_figures=miR_figures,
            populations=populations
            )
    logging.info('Done.')

def _main(argv: Sequence[str]):
    compute(
        data_path=FLAGS.data_path, 
        dataset_name=FLAGS.dataset_name, 
        miR_list=FLAGS.miR_list, 
        cpus=FLAGS.cpus, 
        results_path=FLAGS.results_path, 
        sample_size=FLAGS.sample_size,
        species=FLAGS.species, 
        miR_figures=FLAGS.miR_figures,
        preprocess=FLAGS.preprocess, 
        thresh=FLAGS.thresh,
        populations=FLAGS.populations,
        filter_spots=FLAGS.filter_spots,
        debug=FLAGS.debug)

def main():
    app.run(_main)

if __name__ == '__main__': 
    main()
    