from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "data"

# Path to the project's data files
RAW_DATA = DATA_FOLDER / "desafio_nps_fase_1.csv"
CLEAN_DATA = DATA_FOLDER / "base_nps.parquet"

# Path to the project's model files
MODELS_FOLDER = PROJECT_FOLDER / "models"

# Other project paths
REPORTS_FOLDER = PROJECT_FOLDER / "reports"
IMAGES_FOLDER = REPORTS_FOLDER / "images"