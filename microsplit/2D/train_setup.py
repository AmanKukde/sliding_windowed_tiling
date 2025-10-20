"""
Setup script for the HT_LIF24 dataset using MicroSplit reproducibility code.
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
from microsplit_reproducibility.configs.parameters.HT_LIF24 import get_microsplit_parameters
from microsplit_reproducibility.configs.data.HT_LIF24 import get_data_configs
from microsplit_reproducibility.datasets.HT_LIF24 import get_train_val_data

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
)
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.notebook_utils.HT_LIF24 import (
    ExposureDuration,
    define_experiment_config,
    load_pretrained_model,
)

from careamics.lightning import VAEModule


def setup_environment_htlif24(
    exposure_duration: str = "5ms",
    num_channels: int = 2,
    sliding_window_flag: bool = True,
    evaluate_on_validation_data: bool = True,
    pretrained: bool = True,
):
    """
    Sets up the MicroSplit environment for the HT_LIF24 dataset.

    Args:
        exposure_duration (str): Exposure duration ('2ms', '3ms', '5ms', '20ms', '500ms')
        num_channels (int): Number of channels to use (2, 3, or 4)
        sliding_window_flag (bool): Whether to use sliding window dataloaders
        evaluate_on_validation_data (bool): Use validation data instead of test data
        pretrained (bool): If True, downloads and loads pretrained checkpoint

    Returns:
        model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt
    """
    print(f"🔧 Setting up HT_LIF24 environment (exposure={exposure_duration}, channels={num_channels})")

    # -------------------------------------------------------------------------
    # Step 1: Define experiment configuration (channel indices and exposure)
    # -------------------------------------------------------------------------
    CHANNEL_IDX_LIST, TARGET_CHANNEL_IDX_LIST, EXPOSURE_DURATION = define_experiment_config(
        num_channels=num_channels, exposure=ExposureDuration.Medium
    )

    # -------------------------------------------------------------------------
    # Step 2: Download data and noise models
    # -------------------------------------------------------------------------
    DATA = pooch.create(
        path=Path(f"/group/jug/aman/Datasets/HT_LIF24/"),
        base_url="https://download.fht.org/jug/msplit/ht_lif24/data/",
        registry={f"ht_lif24_{EXPOSURE_DURATION}.zip": None},
    )

    NOISE_MODELS = pooch.create(
        path=Path(f"./noise_models/{EXPOSURE_DURATION}/"),
        base_url=f"https://download.fht.org/jug/msplit/ht_lif24/noise_models/{EXPOSURE_DURATION}/",
        registry={f"noise_model_Ch{ch}.npz": None for ch in TARGET_CHANNEL_IDX_LIST},
    )

    print("📥 Downloading noise models and dataset (if missing)...")
    for fname in NOISE_MODELS.registry:
        NOISE_MODELS.fetch(fname, progressbar=True)

    for fname in DATA.registry:
        DATA.fetch(fname, processor=pooch.Unzip(), progressbar=True)

    # -------------------------------------------------------------------------
    # Step 3: Dataset configuration and creation
    # -------------------------------------------------------------------------
    train_data_config, val_data_config, test_data_config = get_data_configs(
        dset_type=EXPOSURE_DURATION,
        channel_idx_list=CHANNEL_IDX_LIST,
        sliding_window_flag=sliding_window_flag,
    )

    experiment_params = get_microsplit_parameters(
        dset_type=EXPOSURE_DURATION,
        nm_path=NOISE_MODELS.path,
        channel_idx_list=CHANNEL_IDX_LIST,
    )

    datapath = Path(DATA.path) / f"ht_lif24_{EXPOSURE_DURATION}.zip.unzip/{EXPOSURE_DURATION}"

    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=datapath,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )

    # -------------------------------------------------------------------------
    # Step 4: Select evaluation dataset
    # -------------------------------------------------------------------------
    dset = val_dset if evaluate_on_validation_data else test_dset
    print(f"📊 Using {'validation' if evaluate_on_validation_data else 'test'} dataset "
          f"with {dset.get_num_frames()} frames.")

    # -------------------------------------------------------------------------
    # Step 5: Download pretrained model (if requested)
    # -------------------------------------------------------------------------
    if pretrained:
        if len(TARGET_CHANNEL_IDX_LIST) == 2:
            ckpt_name = f"best_{TARGET_CHANNEL_IDX_LIST[0]}_{TARGET_CHANNEL_IDX_LIST[1]}.ckpt"
        elif len(TARGET_CHANNEL_IDX_LIST) == 3:
            ckpt_name = f"best_{'_'.join(map(str, TARGET_CHANNEL_IDX_LIST))}.ckpt"
        elif len(TARGET_CHANNEL_IDX_LIST) == 4:
            ckpt_name = f"best_{'_'.join(map(str, TARGET_CHANNEL_IDX_LIST))}.ckpt"
        else:
            raise ValueError(f"Unsupported number of channels: {len(TARGET_CHANNEL_IDX_LIST)}")

        MODEL_CHECKPOINTS = pooch.create(
            path=Path(f"./pretrained_checkpoints/{EXPOSURE_DURATION}/"),
            base_url=f"https://download.fht.org/jug/msplit/ht_lif24/ckpts/{EXPOSURE_DURATION}/",
            registry={ckpt_name: None},
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
    # Step 6: Build model configuration and instantiate model
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

    if pretrained and selected_ckpt is not None:
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
    model, config, (train_dset, val_dset, test_dset), ckpt = setup_environment_htlif24()
    print("Setup complete. Model and datasets are ready.")
