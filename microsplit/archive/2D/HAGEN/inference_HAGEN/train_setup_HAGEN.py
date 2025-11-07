"""
Setup script for the HAGEN dataset using MicroSplit reproducibility code.
This script:
  - Downloads data and pretrained models (if not available)
  - Creates train/val/test datasets
  - Builds a VAEModule model from configs
  - Loads the pretrained checkpoint

Returns:
    model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt
"""

import os
import platform
from pathlib import Path

import pooch
import tifffile
from careamics.lightning import VAEModule

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config
)
from microsplit_reproducibility.utils.io import load_checkpoint_path, load_checkpoint
from microsplit_reproducibility.utils.utils import plot_input_patches
from microsplit_reproducibility.datasets import create_train_val_datasets

# Dataset specific imports...
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import (
    get_microsplit_parameters,
)
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import load_pretrained_model


def setup_environment_HAGEN(
    sliding_window_flag: bool = False,
):
    """
    Sets up the MicroSplit environment for the HAGEN dataset.

    Args:
        exposure_duration (str): Exposure duration ('2ms', '3ms', '5ms', '20ms', '500ms')
        num_channels (int): Number of channels to use (2, 3, or 4)
        sliding_window_flag (bool): Whether to use sliding window dataloaders
        evaluate_on_validation_data (bool): Use validation data instead of test data
        pretrained (bool): If True, downloads and loads pretrained checkpoint

    Returns:
        model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt
    """
    print(f"🔧 Setting up HAGEN environment")

    # -------------------------------------------------------------------------
    # Step 1: Define experiment configuration (channel indices and exposure)
    # -------------------------------------------------------------------------

    DATA_PATH = Path("/group/jug/aman/Datasets/Hagen/")
    NM_PATH = Path("/group/jug/aman/microsplit_Hagen_26Oct25/noise_models/")
    
    NUM_CHANNELS = 2
    """The number of channels considered for the splitting task."""
    BATCH_SIZE = 32*8
    """The batch size for training."""
    PATCH_SIZE = (64, 64)
    """The size of the patches fed to the network for training in (Y, X)."""
    EPOCHS = 50
    """The number of epochs to train the network."""

    assert len(PATCH_SIZE) == 2, "PATCH_SIZE must be a tuple of length 2 (Y, X) since we are using 2D data."
    
    # -------------------------------------------------------------------------
    # Step 3: Dataset configuration and creation
    # -------------------------------------------------------------------------
# setting up train, validation, and test data configs
    train_data_config, val_data_config, test_data_config = get_data_configs(
        image_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        sliding_window_flag=sliding_window_flag,
    )

    # setting up MicroSplit parametrization
    experiment_params = get_microsplit_parameters(
        algorithm="denoisplit",
        img_size=PATCH_SIZE,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        multiscale_count=3,
        noise_model_path=NM_PATH,
        target_channels=NUM_CHANNELS,
    )

    # create the dataset
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=DATA_PATH,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=val_data_config,
        load_data_func=get_train_val_data,
    )


    # -------------------------------------------------------------------------
    #* Step 6: Build model configuration and instantiate model
    # -------------------------------------------------------------------------
    experiment_params["data_stats"] = data_stats

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

    ckpt_folder = Path("/group/jug/aman/microsplit_Hagen_26Oct25/checkpoints")
    selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)

    if selected_ckpt is not None:
        load_pretrained_model(model, selected_ckpt)
        print(f"✅ Loaded pretrained checkpoint from: {selected_ckpt}")
    else:
        print("⚠️ Model created without pretrained weights.")

    model.eval()
    model.cuda()

    print("✅ Environment setup complete.")
    return model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    model, config, (train_dset, val_dset, test_dset), ckpt_path = setup_environment_HAGEN()
    print("Setup complete. Model and datasets are ready.")