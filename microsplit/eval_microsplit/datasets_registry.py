# dataset_registry.py
import os
from pathlib import Path
from setup_dataloaders import (
    setup_dataset_PAVIA_ATN,
    setup_dataset_HAGEN,
    setup_dataset_HT_LIF24,
    setup_dataset_HT_H24
)

# Allow flexible environment path for data storage
DEFAULT_DATA_ROOT = Path(os.getenv("MSPLIT_DATA_ROOT", "/group/jug/aman/Datasets"))

DATASET_PATHS = {
    "PAVIA_ATN": DEFAULT_DATA_ROOT / "PAVIA_ATN",
    "HAGEN": DEFAULT_DATA_ROOT / "HAGEN",
    "HT_LIF24": DEFAULT_DATA_ROOT / "HT_LIF24",
    "HT_H24":DEFAULT_DATA_ROOT/"HT_H24"
}

DATASET_SETUP_FUNCS = {
    "PAVIA_ATN": setup_dataset_PAVIA_ATN,
    "HAGEN": setup_dataset_HAGEN,
    "HT_LIF24": setup_dataset_HT_LIF24,
    "HT_H24":setup_dataset_HT_H24
}
