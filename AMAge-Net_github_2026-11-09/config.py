"""Data locations. Edit the paths below, or put the same dictionary in `config_local.py` (git-ignored),
which overrides this file if present.

Expected layout per cohort (one folder per participant):
  T1w volume (preprocessed, MNI space)  -> DATASETS[cohort]['t1'].format(sid=...)
  90x90 AAL functional connectivity CSV -> DATASETS[cohort]['fc'].format(sid=...)   (row/column labels in the first row/column)
  age table (columns: Subject, Age)     -> DATASETS[cohort]['ages']
  sex table (columns: Subject, Sex with values MALE/FEMALE; optional, Cam-CAN only) -> DATASETS[cohort]['sex']
"""
DATASETS = {
    'camcan': dict(
        t1='/path/to/CamCAN/Cam-CAN_T1W/{sid}/T1.nii.gz',
        fc='/path/to/CamCAN/timeseries/{sid}/connectome.csv',
        ages='/path/to/CamCAN/CamCAN_Age.csv',
        sex='/path/to/CamCAN/CamCAN_Age_Sex_Gender.csv',
    ),
    'oasis': dict(
        t1='/path/to/OASIS-3/CN/{sid}/brain.nii.gz',
        fc='/path/to/OASIS-3/CN/{sid}/connectome.csv',
        ages='/path/to/OASIS-3/OASIS3_CN.csv',
        sex=None,
    ),
}

# AAL atlas (SPM12 version, 2 mm MNI) used for regional saliency
AAL_NII = '/path/to/aal_for_SPM12/atlas/AAL.nii'
AAL_LABELS = '/path/to/aal_for_SPM12/ROI_MNI_V4.txt'

RESULTS_DIR = 'results'   # per-model, per-fold predictions / checkpoints / splits
CACHE_DIR = 'cache'       # resampled T1w volumes (.npy), created on first use
ANALYSIS_DIR = 'analysis_out'

try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
