# Welcome to the miTEA-HiRes package!

miTEA-HiRes is an open-source package, designed to easily compute high resolution microRNA activity 
maps. This package is used in our paper "Inferring single-cell and spatial microRNA 
activity from transcriptomics data" by Herbst et al [1].

If you use miTEA-HiRes in your research, please cite [1]. 
*** add link and publication 

## Installation
*** Update after pip-install is ready
You will need Python 3.8 or above.
To install the package use the following command:
    pip install mitea-hires

*** Remove after done:
Requirements:
python >= 3.8
absl (pip3.8 install absl-py)
anndata (pip3.8 install anndata)
pandas (pip3.8 install pandas)
numpy==1.23 (bug in xlmhg with np.float)
matplotlib (pip3.8 install matplotlib)
scanpy (pip3.8 install scanpy)
scipy (pip3.8 install scipy)
tqdm (pip3.8 install tqdm)
xlmhg (pip3.8 install xlmhg)

*** Sharon - do I need to check if the libraries could be newer? should I mention the versions even if they were not important?
*** check on a new environment that it works well with the libraries dependencies.
*** check if the newest python can be used. 

## Support and bug report:
This package is provided by Yakhini research group.
If you encounter any issues, please contact efiherbst through gmail.

# Example 1 - Single-cell in 'Total' activity mode
In this case, miTEA-HiRes computes the activity for each cell and microRNA, and produces activity 
maps on a UMAP layout for the most active microRNAs.
## Input format:
One or more raw counts matrices should be available in the data path, in the form of zipped or 
unzipped 'txt', 'tsv', 'mtx' or 'pkl' files. 
 
 ```
YOUR_DATA_FOLDER
|   counts_1.txt
│   counts_2.txt
|   ...
```

If 10X files (i.e. 'mtx') are used for single-cell data, your data folder should look as follows: 
 ```
YOUR_DATA_FOLDER
|   *_barcodes.tsv.gz   
│   *_genes.tsv.gz
│   *.mtx.gz
```
## Execution:
You may use the package via commandline:
```
python3.8 main.py -data_path='PATH_TO_YOUR_DATA' -dataset_name='DATASET_NAME'
```

Or by importing the library into your code:
```
import mitea-hires as mhr

data_path = 'PATH_TO_YOUR_DATA'
dataset_name = 'NAME_OF_YOUR_DATASET' 

mhr.compute(data_path=data_path, dataset_name=dataset_name)
```

# Example 2 - Single-cell in 'Comaprative' activity mode
In this case, miTEA-HiRes computes microRNA differential activity for two populations of interest. 
microRNA activity for each population is presented on histogram and UMAP layouts for miroRNAs of 
potential interest.
## Input format:
Same as in Example 1, with the following required preprocessing:
A unique population string should be included withing each cell id string.
Taking for example one cell from 'CONTROL' population and one from 'DISEASE':
1. Cell string 'AACAATGTGCTCCGAG' should be transformed to 'AACAATGTGCTCCGAG_CONTROL'.
2. Cell string 'AACAGCCTCCTGACTA' should be transformed to 'AACAGCCTCCTGACTA_DISEASE'.
## Execution:
Example using commandline:
```
python3.8 main.py -data_path='PATH_TO_YOUR_DATA' -dataset_name='DATASET_NAME' -populations='DISEASE','CONTROL'
```
# Example 3 - Spatial trascriptomics
In this case, miTEA-HiRes computes the activity for each spot and microRNA, and produces spatial 
activity maps for microRNAs which are most abundantly active throughout the spots.
## Input format:
Make sure you have obtained the following files for your Visium data (these can be downloaded 
from Visium website):
> filtered_feature_bc_matrix
>
> spatial
Your data path should then look as follows:
 ```
YOUR_DATA_FOLDER
└───filtered_feature_bc_matrix
│   │   barcodes.tsv.gz
│   │   features.tsv.gz
│   │   matrix.mtx.gz
└───spatial
	│   tissue_positions_list.csv
	│   ...
```
## Execution:
```
python3.8 main.py -data_path='PATH_TO_YOUR_DATA' -dataset_name='DATASET_NAME'
```

# Outputs
'results' folder is generated under the provided data_path unless specified otherwise and contains:
1. csv files with the activity p-values (and other related statistics) for every cell/spot and 
    every microRNA.
2. html file with sorted microRNAs according to their potential interest, including links to 
    activity maps.
3. folder with activity map plots.

# Recommended setup
mitea-hires is desinged to utilize all CPUs available for parallel computing, hence in order to 
speed up the processing time, you may want to consider using resources with more CPUs.
For example, an input of spatial trascriptomics data including 2,264 spots, ~6,000 genes per spot, 
computing activity for 706 microRNAs using a cloud instance with 16 cores takes 22 minutes.

# Advanced usage
run ```mitea-hires --help``` in command line to see additional supported flags.
###CHECK THIS

mitea-hires can also be imported within your python script and then, end-to-end compuation can be 
executed calling the 'compute' function. Alternatively, in order to use parts of the computation, 
other functions can be called (see docstrings).

## Supported species
miTEA-HiRes currently supports mouse and human microRNAs. 

[1] Inferring single-cell and spatial microRNA activity from transcriptomics data. *Herbst et al.* 
