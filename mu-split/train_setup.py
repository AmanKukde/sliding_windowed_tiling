# %%
from pathlib import Path
import os
import glob
import torch
import ml_collections
import numpy as np

from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import get_microsplit_parameters
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data

from usplit.training import create_model
from usplit.config_utils import load_config
from usplit.core.data_type import DataType
from usplit.core.data_split_type import DataSplitType
from usplit.core.model_type import ModelType

torch.multiprocessing.set_sharing_strategy('file_system')


# %%
def get_config_file_from_ckpt_dir(ckpt_dir: str) -> str:
    """Automatically find the config.pkl inside a checkpoint directory."""
    pkl_files = glob.glob(os.path.join(ckpt_dir, "*.pkl"))
    assert len(pkl_files) == 1, f"Expected exactly one .pkl file in {ckpt_dir}, found: {pkl_files}"
    return pkl_files[0]


def get_best_checkpoint(ckpt_dir: str) -> str:
    """Find the best checkpoint (_best.ckpt) in a folder."""
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*_best.ckpt"))
    assert len(ckpt_files) == 1, f"Expected exactly one _best.ckpt in {ckpt_dir}, found: {ckpt_files}"
    return ckpt_files[0]


# %%
def setup_environment(dataset_name, lc_type, modality,
                      img_sz=64, sliding_window_flag=False, batch_size=32, num_workers=6):

    DATA_PATH = Path(f"/group/jug/aman/Datasets/{dataset_name}")
    ckpt_dir = f"/home/aman.kukde/Projects/models/paper_models/{dataset_name}/{modality}/{lc_type}/"
    NM_PATH = Path("./noise_models/")

    # Data configs
    multiscale_lowres_count = 5
    train_data_config, val_data_config, test_data_config = get_data_configs(
        image_size=(img_sz, img_sz),
        num_channels=2,
        sliding_window_flag=sliding_window_flag,
        multiscale_lowres_count=multiscale_lowres_count,
    )

    # # MicroSplit experiment params (optional, in case used later)
    # experiment_params = get_microsplit_parameters(
    #     algorithm="musplit",
    #     img_size=(img_sz, img_sz),
    #     batch_size=batch_size,
    #     num_epochs=50,
    #     multiscale_count=multiscale_lowres_count,
    #     noise_model_path=NM_PATH,
    #     target_channels=2,
    # )

    # Create datasets
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=DATA_PATH,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )

    
    config_fpath = get_config_file_from_ckpt_dir(ckpt_dir)
    config = ml_collections.ConfigDict(load_config(config_fpath))
    old_image_size = None

    # Patch config if necessary
    with config.unlocked():
        if 'test_fraction' not in config.training:
            config.training.test_fraction = 0.0
        if 'datadir' not in config:
            config.datadir = ''
        if 'encoder' not in config.model:
            config.model.encoder = ml_collections.ConfigDict()
            config.model.decoder = ml_collections.ConfigDict()
            config.model.encoder.dropout = config.model.dropout
            config.model.decoder.dropout = config.model.dropout
            config.model.encoder.blocks_per_layer = config.model.blocks_per_layer
            config.model.decoder.blocks_per_layer = config.model.blocks_per_layer
            config.model.encoder.n_filters = config.model.n_filters
            config.model.decoder.n_filters = config.model.n_filters
        if 'multiscale_retain_spatial_dims' not in config.model.decoder:
            config.model.decoder.multiscale_retain_spatial_dims = False
        if 'res_block_kernel' not in config.model.encoder:
            config.model.encoder.res_block_kernel = 3
            config.model.decoder.res_block_kernel = 3
        if 'res_block_skip_padding' not in config.model.encoder:
            config.model.encoder.res_block_skip_padding = False
            config.model.decoder.res_block_skip_padding = False
        if config.data.data_type == DataType.CustomSinosoid:
            if 'max_vshift_factor' not in config.data:
                config.data.max_vshift_factor = config.data.max_shift_factor
                config.data.max_hshift_factor = 0
            if 'encourage_non_overlap_single_channel' not in config.data:
                config.data.encourage_non_overlap_single_channel = False
        if 'skip_bottom_layers_count' in config.model:
            config.model.skip_bottom_layers_count = 0
        if 'logvar_lowerbound' not in config.model:
            config.model.logvar_lowerbound = None
        if 'train_aug_rotate' not in config.data:
            config.data.train_aug_rotate = False
        if 'multiscale_lowres_separate_branch' not in config.model:
            config.model.multiscale_lowres_separate_branch = False
        if 'multiscale_retain_spatial_dims' not in config.model:
            config.model.multiscale_retain_spatial_dims = False
        config.data.train_aug_rotate = False
        if 'randomized_channels' not in config.data:
            config.data.randomized_channels = False
        if 'predict_logvar' not in config.model:
            config.model.predict_logvar = None
        if 'batchnorm' in config.model and 'batchnorm' not in config.model.encoder:
            config.model.decoder.batchnorm = config.model.batchnorm
            config.model.encoder.batchnorm = config.model.batchnorm
        if 'conv2d_bias' not in config.model.decoder:
            config.model.decoder.conv2d_bias = True
        if img_sz is not None:
            old_image_size = config.data.image_size
            config.data.image_size = img_sz
        config.model.mode_pred = True
        if config.model.model_type == ModelType.UNet and 'n_levels' not in config.model:
            config.model.n_levels = 4
        if config.model.model_type == ModelType.UNet and 'init_channel_count' not in config.model:
            config.model.init_channel_count = 64
        if 'skip_receptive_field_loss_tokens' not in config.loss:
            config.loss.skip_receptive_field_loss_tokens = []

    if lc_type =="regularLC":
        config.model.z_dims = [128, 128, 128, 128]
        config.model.multiscale_retain_spatial_dims = True
        config.model.decoder.multiscale_retain_spatial_dims = True
    elif lc_type == "LeanLC":
        config.model.z_dims = [128, 128, 128, 128]
        config.model.multiscale_retain_spatial_dims = False
        config.model.decoder.multiscale_retain_spatial_dims = False
    elif lc_type == "DeepLC":
        config.model.z_dims = [128, 128, 128, 128, 128, 128, 128, 128]
        config.model.multiscale_retain_spatial_dims = True
        config.model.decoder.multiscale_retain_spatial_dims = True

    # Compute mean/std for normalization
    if config.data.target_separate_normalization:
        mean_fr_model, std_fr_model = train_dset.compute_individual_mean_std()
    else:
        mean_fr_model, std_fr_model = train_dset.get_mean_std()

    # Build model + load checkpoint
    model = create_model(config, mean_fr_model, std_fr_model)
    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath, weights_only=False)
    print(f"Loading checkpoint from {ckpt_fpath}")
    _ = model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()
    model.set_params_to_same_device_as(torch.Tensor(1).cuda())

    print('Loaded checkpoint from epoch', checkpoint['epoch'])

    # Restore old image size if needed
    with config.unlocked():
        if old_image_size is not None:
            config.data.image_size = old_image_size

    return model, config, (train_dset, val_dset, test_dset), ckpt_dir


# %%
if __name__ == "__main__":
    model, config, (train_dset, val_dset, test_dset), ckpt_dir = setup_environment()
    print("Setup complete. Model and datasets are ready.")
