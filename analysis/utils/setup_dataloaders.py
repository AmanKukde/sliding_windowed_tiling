
from pathlib import Path
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import get_microsplit_parameters
from microsplit_reproducibility.datasets import create_train_val_datasets

import pooch
def setup_dataset_usplit(dataset_name, img_sz=64, sliding_window_flag=False):
    if dataset_name.upper() == "PAVIA_ATN":
        DATA_PATH = Path("/group/jug/aman/Datasets/PAVIA_ATN")
    elif dataset_name.upper() == "HAGEN":
        DATA_PATH = Path("/group/jug/aman/Datasets/Hagen")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_cfg, val_cfg, test_cfg = get_data_configs(
        image_size=(img_sz, img_sz),
        num_channels=2,
        sliding_window_flag=sliding_window_flag,
        multiscale_lowres_count=5
    )

    params = get_microsplit_parameters(
        algorithm="musplit",
        img_size=(img_sz, img_sz),
        batch_size=32,
        num_epochs=10,
        multiscale_count=5,
        noise_model_path=Path("./noise_models/"),
        target_channels=2
    )

    train_dset, val_dset, test_dset, _ = create_train_val_datasets(
        datapath=DATA_PATH,
        train_config=train_cfg,
        val_config=val_cfg,
        test_config=test_cfg,
        load_data_func=get_train_val_data
    )

    return test_dset


def setup_dataset_microsplit_HT_LIF24(
    exposure_duration: str = "5ms",
    num_channels: int = 2,
    sliding_window_flag: bool = False,
):
    from microsplit_reproducibility.configs.parameters.HT_LIF24 import get_microsplit_parameters
    from microsplit_reproducibility.configs.data.HT_LIF24 import get_data_configs
    from microsplit_reproducibility.datasets.HT_LIF24 import get_train_val_data
    from microsplit_reproducibility.notebook_utils.HT_LIF24 import ExposureDuration, define_experiment_config

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
    
    return test_dset
    

def setup_dataset_microsplit_HT_H24(sliding_window_flag=False):
    from microsplit_reproducibility.configs.parameters.HT_H24 import get_microsplit_parameters
    from microsplit_reproducibility.configs.data.HT_H24 import get_data_configs
    from microsplit_reproducibility.datasets.HT_H24 import get_train_val_data

    print(f"🔧 Setting up HT_H24 environment")

    DATA = pooch.create(
    path="/group/jug/aman/Datasets/HT_H24/data",
    base_url="https://download.fht.org/jug/msplit/ht_h24/data",
    registry={f"ht_h24.zip": None},
    )

    NOISE_MODELS = pooch.create(
        path=f"/group/jug/aman/Datasets/HT_H24/noise_models/",
        base_url=f"https://download.fht.org/jug/msplit/ht_h24/noise_models/",
        registry={
            f"noise_model_Ch0.npz": None,
            f"noise_model_Ch1.npz": None,
        },
    )
    for fname in NOISE_MODELS.registry:
        NOISE_MODELS.fetch(fname, progressbar=True)
    print('---------')
    for fname in DATA.registry:
        DATA.fetch(fname, processor=pooch.Unzip(), progressbar=True)
    
    # setting up train, validation, and test data configs
    train_data_config, val_data_config, test_data_config = get_data_configs(sliding_window_flag = sliding_window_flag)
    # setting up MicroSplit parametrization
    experiment_params = get_microsplit_parameters(nm_path=NOISE_MODELS.path, batch_size=32)

    # start the download of required files
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=DATA.path / f"ht_h24.zip.unzip/ht_h24",
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=val_data_config,
        load_data_func=get_train_val_data,
    )
    
    return test_dset