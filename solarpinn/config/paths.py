from pathlib import Path

#Project Root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#Data

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

INTERIM_DATA_DIR = DATA_DIR / "interim"

CACHE_DIR = DATA_DIR / "cache"

MAGNETOGRAM_DIR = DATA_DIR / "magnetograms"

SUNSPOT_DIR = DATA_DIR / "sunspots"

FLARE_DIR = DATA_DIR / "flare_catalogs"

#Results

RESULT_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = RESULT_DIR / "figures"

MODEL_DIR = RESULT_DIR / "models"

METRIC_DIR = RESULT_DIR / "metrics"

PREDICTION_DIR = RESULT_DIR / "predictions"

#Checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

#Logs
LOG_DIR = PROJECT_ROOT / "logs"

#Documentation
DOCS_DIR = PROJECT_ROOT / "docs"

#Notebooks
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

#Scripts
SCRIPT_DIR = PROJECT_ROOT / "scripts"

#Experiments
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

FORECAST_DIR = EXPERIMENT_DIR / "forecasting"

RECONSTRUCTION_DIR = EXPERIMENT_DIR / "reconstruction"

FLARE_PREDICTION_DIR = EXPERIMENT_DIR / "flare_prediction"



