
from pathlib import Path

from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import load_pretrained_model
import pooch
from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config
)
from careamics.lightning import VAEModule
from microsplit_reproducibility.utils.io import load_checkpoint_path

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

    # return model, experiment_config, (train_dset, val_dset, test_dset), selected_ckpt
    return test_dset


def setup_dataset_HT_LIF24(
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
        path=Path(f"/group/jug/aman/Datasets/HT_LIF24/data"),
        base_url="https://download.fht.org/jug/msplit/ht_lif24/data/",
        registry={f"ht_lif24_{EXPOSURE_DURATION}.zip": None},
    )

    NOISE_MODELS = pooch.create(
        path=Path(f"/group/jug/aman/Datasets/HT_LIF24/noise_models/{EXPOSURE_DURATION}/"),
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
    if len(TARGET_CHANNEL_IDX_LIST) == 2:
        ckpt_name = f"best_{TARGET_CHANNEL_IDX_LIST[0]}_{TARGET_CHANNEL_IDX_LIST[1]}.ckpt"
    elif len(TARGET_CHANNEL_IDX_LIST) == 3:
        ckpt_name = f"best_{TARGET_CHANNEL_IDX_LIST[0]}_{TARGET_CHANNEL_IDX_LIST[1]}_{TARGET_CHANNEL_IDX_LIST[2]}.ckpt"
    elif len(TARGET_CHANNEL_IDX_LIST) == 4:
        ckpt_name = f"best_{TARGET_CHANNEL_IDX_LIST[0]}_{TARGET_CHANNEL_IDX_LIST[1]}_{TARGET_CHANNEL_IDX_LIST[2]}_{TARGET_CHANNEL_IDX_LIST[3]}.ckpt"
    else:
        raise ValueError(f"Unsupported number of channels: {len(TARGET_CHANNEL_IDX_LIST)}")

    MODEL_CHECKPOINTS = pooch.create(
        path=f"/group/jug/aman/Datasets/HT_LIF24/pretrained_checkpoints/{EXPOSURE_DURATION}/",
        base_url=f"https://download.fht.org/jug/msplit/ht_lif24/ckpts/{EXPOSURE_DURATION}",
        registry={ckpt_name: None},
    )

    pretrained_model_available = False
    for f in MODEL_CHECKPOINTS.registry:
        if MODEL_CHECKPOINTS.is_available(f):
            MODEL_CHECKPOINTS.fetch(f"{f}", progressbar=True)
            pretrained_model_available = True

    if not pretrained_model_available:
        print("No suitable pretrained model for your data seems to be available.\n"
            "Please train the model using the notebook '01_train.ipynb' or download \n"
            "correct notebook. If multiple checkpoints are present in the folder, \n"
            "remove the ones that you won't be using")

    selected_ckpt = load_checkpoint_path(str('/group/jug/aman/Datasets/HT_LIF24/pretrained_checkpoints/5ms'), best=True)
    
    # making our data_stas known to the experiment (model) we prepare
    experiment_params["data_stats"] = data_stats

    # setting up model config (using default parameters)
    model_config = get_model_config(**experiment_params)

    # NOTE: The creation of the following configs are not strictly necessary for prediction,
    #     but they ARE currently expected by the create_algorithm_config function below.
    #     They act as a placeholder for now and we will work to remove them in a following release
    loss_config = get_loss_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
        **experiment_params
    )

    # finally, assemble the full set of experiment configurations...
    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
    )  
    
    
    model = VAEModule(algorithm_config=experiment_config)
    from microsplit_reproducibility.notebook_utils.HT_LIF24 import load_pretrained_model
    load_pretrained_model(model, selected_ckpt)
    return model, experiment_config, train_dset, val_dset, test_dset
    
def setup_dataset_HT_H24(sliding_window_flag=False):
    
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
    MODEL_CHECKPOINTS = pooch.create(
    path=f"/group/jug/aman/Datasets/HT_H24/pretrained_checkpoints/",
    base_url=f"https://download.fht.org/jug/msplit/ht_h24/ckpts/",
    registry={f"best.ckpt": None},
)

    pretrained_model_available = False
    for f in MODEL_CHECKPOINTS.registry:
        if MODEL_CHECKPOINTS.is_available(f):
            MODEL_CHECKPOINTS.fetch(f"{f}", progressbar=True)
            pretrained_model_available = True

    assert pretrained_model_available, "No suitable pretrained model for your data seems to be available.\nPlease train the model using the notebook '01_train.ipynb'."
    ckpt_folder = '/group/jug/aman/Datasets/HT_H24/pretrained_checkpoints/'

    selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)
    
    experiment_params["data_stats"] = data_stats

    # setting up model config (using default parameters)
    model_config = get_model_config(**experiment_params)
    # making our data_stats known to the experiment (model) we prepare
    experiment_params["data_stats"] = data_stats

    # setting up model config (using default parameters)
    model_config = get_model_config(**experiment_params)

    # NOTE: The creation of the following configs are not strictly necessary for prediction,
    #     but they ARE currently expected by the create_algorithm_config function below.
    #     They act as a placeholder for now and we will work to remove them in a following release
    loss_config = get_loss_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
        **experiment_params
    )

    # finally, assemble the full set of experiment configurations...
    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
    )
    model = VAEModule(algorithm_config=experiment_config)
    return model, experiment_config,train_dset, val_dset, test_dset

def setup_dataset_PAVIA_ATN(
    sliding_window_flag: bool = False,
):
    """
    Sets up the MicroSplit environment for the PAVIA_ATN dataset.

    Args:
        exposure_duration (str): Exposure duration ('2ms', '3ms', '5ms', '20ms', '500ms')
        num_channels (int): Number of channels to use (2, 3, or 4)
        sliding_window_flag (bool): Whether to use sliding window dataloaders
        evaluate_on_validation_data (bool): Use validation data instead of test data
        pretrained (bool): If True, downloads and loads pretrained checkpoint

    Returns:
        model, experiment_config, train_dset, val_dset, test_dset
    
    """
    print(f"🔧 Setting up PAVIA_ATN environment")
    from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
    from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data
    from microsplit_reproducibility.configs.parameters.custom_dataset_2D import get_microsplit_parameters
    # -------------------------------------------------------------------------
    # Step 1: Define experiment configuration (channel indices and exposure)
    # -------------------------------------------------------------------------

    DATA_PATH = Path("/group/jug/aman/Datasets/PAVIA_ATN/data")
    NM_PATH = Path("/group/jug/aman/Datasets/PAVIA_ATN/noise_models/")
    
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

    ckpt_folder = Path("/group/jug/aman/Datasets/PAVIA_ATN/model_checkpoints")
    selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)
   

    if selected_ckpt is not None:
        load_pretrained_model(model, selected_ckpt)
        print(f"✅ Loaded pretrained checkpoint from: {selected_ckpt}")
    else:
        print("⚠️ Model created without pretrained weights.")

    model.eval()
    model.cpu()

    print("✅ Environment setup complete.")
    return model, experiment_config, train_dset, val_dset, test_dset

def setup_dataset_HAGEN(
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
    from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
    from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data
    from microsplit_reproducibility.configs.parameters.custom_dataset_2D import get_microsplit_parameters
    # -------------------------------------------------------------------------
    # Step 1: Define experiment configuration (channel indices and exposure)
    # -------------------------------------------------------------------------

    DATA_PATH = Path("/group/jug/aman/Datasets/HAGEN/data")
    NM_PATH = Path("/group/jug/aman/Datasets/HAGEN/noise_models/")
    
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

    ckpt_folder = Path("/group/jug/aman/Datasets/HAGEN/model_checkpoints")
    selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)

    if selected_ckpt is not None:
        load_pretrained_model(model, selected_ckpt)
        print(f"✅ Loaded pretrained checkpoint from: {selected_ckpt}")
    else:
        print("⚠️ Model created without pretrained weights.")

    model.eval()
    model.cuda()

    print("✅ Environment setup complete.")
    return model, experiment_config, train_dset, val_dset, test_dset

# main_inference.py
from pathlib import Path
import argparse
from inference_utils import (
    run_inference_original,
    run_inference_sliding_and_stitch,
)
from dataset_registry import DATASET_SETUP_FUNCS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MicroSplit LVAE inference")

    # -------------------------------------------------------------------------
    # Dataset and model setup
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PAVIA_ATN", "HAGEN", "HT_LIF24"],
        help="Dataset to run inference on"
    )
    parser.add_argument("--num_channels", type=int, default=2, help="Number of channels (2, 3, or 4)")
    parser.add_argument("--exposure_duration", type=str, default="5ms", help="Exposure duration for HT_LIF24 dataset")

    # -------------------------------------------------------------------------
    # Inference behavior
    # -------------------------------------------------------------------------
    parser.add_argument("--reduce_dataset_size", action="store_true",
                        help="Reduce dataset size for quick testing (val/test only)")
    parser.add_argument("--sliding_window_flag", action="store_true",
                        help="Enable sliding-window inference")
    parser.add_argument("--stitch_only", action="store_true",
                        help="Stitch previously saved tiles only")
    parser.add_argument("--results_root", type=str, default="./Microsplit_predictions",
                        help="Where to save results")
    parser.add_argument("--raw_preds_dir", type=str, default=None,
                        help="Directory with raw .npy tiles (for stitch_only mode)")

    # -------------------------------------------------------------------------
    # Performance and configuration parameters
    # -------------------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mmse_count", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--load_pretrained_ckpt", action="store_false",
                        help="Skip pretrained model loading")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Dataset setup
    # -------------------------------------------------------------------------
    setup_fn = DATASET_SETUP_FUNCS.get(args.dataset.upper())
    if setup_fn is None:
        raise ValueError(f"❌ Unknown dataset: {args.dataset}")

    setup_kwargs = {
        "sliding_window_flag": args.sliding_window_flag
    }

    # Add dataset-specific extras
    if args.dataset.upper() == "HT_LIF24":
        setup_kwargs["num_channels"] = args.num_channels
        setup_kwargs["exposure_duration"] = args.exposure_duration

    model, experiment_config, train_dset, val_dset, test_dset = setup_fn(**setup_kwargs)

    # -------------------------------------------------------------------------
    # Optional: Reduce dataset for quick testing
    # -------------------------------------------------------------------------
    if args.reduce_dataset_size and hasattr(test_dset, "reduce_data"):
        print("⚙️ Reducing test dataset size for quick testing...")
        test_dset.reduce_data([0])

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    if args.stitch_only:
        print("🧵 Stitching previously saved tiles only...")
        # You’ll implement this
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            results_root=args.results_root,
            batch_size=args.batch_size,
            channels=args.num_channels,
        )

    elif args.sliding_window_flag:
        print("🪟 Running sliding-window inference + stitching...")
        run_inference_sliding_and_stitch(
            model=model,
            train_dset=train_dset,
            test_dset=test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root,
            dataset=args.dataset,
        )

    else:
        print("🎯 Running standard inference...")
        run_inference_original(
            model=model,
            experiment_params=experiment_config,
            test_dset=test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mmse_count=args.mmse_count,
            grid_size=args.grid_size,
            dataset=args.dataset,
            results_root=Path(args.results_root),
        )
