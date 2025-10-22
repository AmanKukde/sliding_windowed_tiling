"""
Setup script for the HT_H23B dataset using MicroSplit reproducibility code.
This script:
  - Downloads data and pretrained models (if not available)
  - Creates train/val/test datasets
  - Builds a VAEModule model from configs
  - Loads the pretrained checkpoint

Returns:
    model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt
"""

from pathlib import Path
import platform
import pooch
import torch
import dill
import os
import numpy as np

from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.configs.parameters.HT_H23B import get_microsplit_parameters
from microsplit_reproducibility.configs.data.HT_H23B import get_data_configs
from microsplit_reproducibility.datasets.HT_H23B import get_test_data

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
)
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.notebook_utils.HT_H23B import load_pretrained_model

from careamics.lightning import VAEModule


def get_num_workers():
    """Utility function to set num_workers based on OS."""
    if platform.system() in ["Windows", "Darwin"]:
        return 0
    else:
        return 3


def setup_environment_HT_H23B(
    test_frame_idx: int = 8,
    sliding_window_flag: bool = True,
    pretrained: bool = True,
):
    """
    Sets up the MicroSplit environment for the HT_H23B dataset.

    Args:
        test_frame_idx (int): Frame index to be used for testing.
        sliding_window_flag (bool): Whether to use sliding window dataloaders.
        pretrained (bool): If True, downloads and loads pretrained checkpoint.

    Returns:
        model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt
    """
    print(f"🔧 Setting up HT_H23B environment (sliding_window={sliding_window_flag})")

    # -------------------------------------------------------------------------
    # Step 1: Download dataset and noise model
    # -------------------------------------------------------------------------
    DATA = pooch.create(
        path=Path("/group/jug/aman/Datasets/HT_H23B/"),
        base_url="https://download.fht.org/jug/msplit/ht_h23b/data/",
        registry={"test.zip": None},
    )

    NOISE_MODELS = pooch.create(
        path=Path("./noise_models/"),
        base_url="https://download.fht.org/jug/msplit/ht_h23b/noise_models/",
        registry={"ht_h23b_nm_raw_data.npz": None},
    )

    print("📥 Downloading noise models and dataset (if missing)...")
    for fname in NOISE_MODELS.registry:
        NOISE_MODELS.fetch(fname, progressbar=True)

    for fname in DATA.registry:
        DATA.fetch(fname, processor=pooch.Unzip(), progressbar=True)

    # -------------------------------------------------------------------------
    # Step 2: Dataset configuration
    # -------------------------------------------------------------------------
    train_data_config, val_data_config, test_data_config = get_data_configs(
        test_frame_idx=test_frame_idx,
        sliding_window_flag=sliding_window_flag,
    )

    experiment_params = get_microsplit_parameters(
        nm_path=Path(NOISE_MODELS.path) / "ht_h23b_nm_raw_data.npz",
        batch_size=64,
    )

    datapath = Path(DATA.path) / "test.zip.unzip"

    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=datapath,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_test_data,
    )

    experiment_params["data_stats"] = data_stats
    experiment_params["num_workers"] = get_num_workers()

    print(f"📊 Dataset created successfully — test frame index: {test_frame_idx}")

    # -------------------------------------------------------------------------
    # Step 3: Download pretrained model
    # -------------------------------------------------------------------------
    if pretrained:
        MODEL_CHECKPOINTS = pooch.create(
            path=Path("./pretrained_checkpoints/"),
            base_url="https://download.fht.org/jug/msplit/ht_h23b/ckpts/",
            registry={"best.ckpt": None},
        )

        print("📥 Downloading pretrained checkpoint (if available)...")
        for f in MODEL_CHECKPOINTS.registry:
            MODEL_CHECKPOINTS.fetch(f, progressbar=True)

        ckpt_folder = Path(MODEL_CHECKPOINTS.path)
        selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)
    else:
        selected_ckpt = None
        print("⚠️ Skipping pretrained checkpoint download (pretrained=False).")

    # -------------------------------------------------------------------------
    # Step 4: Build model configuration and instantiate model
    # -------------------------------------------------------------------------
    model_config = get_model_config(**experiment_params)
    loss_config = get_loss_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(**experiment_params)

    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
    )

    model = VAEModule(algorithm_config=experiment_config)

    if pretrained and selected_ckpt is not None:
        load_pretrained_model(model, selected_ckpt)
        print(f"✅ Loaded pretrained checkpoint from: {selected_ckpt}")
    else:
        print("⚠️ Model created without pretrained weights.")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    print("✅ Environment setup complete.")
    return model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    model, config, (train_dset, val_dset, test_dset), ckpt_path = setup_environment_HT_H23B()
    print("Setup complete. Model and datasets are ready.")
