import numpy as np
from skimage.io import imread, imsave
from pathlib import Path

def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    return imread(path, plugin='tifffile')


def save_tiff(path, data):
    imsave(path, data, plugin='tifffile')


def load_tiffs(paths):
    data = [load_tiff(path) for path in paths]
    return np.concatenate(data, axis=0)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_save_dir(results_root, dataset):
    """
    Build save path as results_root/dataset/
    """
    save_dir = Path(results_root) / dataset 
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_raw_preds_dir(save_dir):
    """
    Build raw predictions folder under save_dir/raw_predictions_<exposure>_<checkpoint>
    """
    raw_dir = save_dir / f"raw_predictions_windowed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir