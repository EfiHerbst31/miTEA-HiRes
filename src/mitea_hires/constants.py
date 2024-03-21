# Supported species.
_SPECIES_HOMO_SAPIENS = 'homo_sapiens'
_SPECIES_MUS_MUSCULUS = 'mus_musculus'
_SUPPORTED_SPECIES = frozenset({
  _SPECIES_HOMO_SAPIENS,
  _SPECIES_MUS_MUSCULUS
})

# Supported options for which microRNAs to select for ploting.
_DRAW_ALL = 'all'
_DRAW_BOTTOM_10 = 'bottom_10'
_DRAW_TOP_10 = 'top_10'
_DRAW_TOP_100 = 'top_100'
_SUPPORTED_DRAW = frozenset({
  _DRAW_ALL,
  _DRAW_BOTTOM_10,
  _DRAW_TOP_10,
  _DRAW_TOP_100
})

# MicroRNA prefixes for the supported species.
_HOMO_SAPIENS_PREFIX = 'hsa'
_MUS_MUSCULUS_PREFIX = 'mmu'

# Supported data types.
_DATA_TYPE_SPATIAL = 'spatial'
_DATA_TYPE_SINGLE_CELL = 'scRNAseq'

# Spatial related folders.
_SPATIAL_FOLDER_1 = 'spatial'
_SPATIAL_FOLDER_2 = 'filtered_feature_bc_matrix'

_SPATIAL_FOLDERS = frozenset({
 _SPATIAL_FOLDER_1,
 _SPATIAL_FOLDER_2
})

# Default threshold used to determine if a microRNA is active or not.
_ACTIVITY_THRESH = 0.00001

# Default sample size used to sample single-cell data.
_MAX_COLS = 10000