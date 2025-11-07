# %%
import os
from pathlib import Path
import re
import dill
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import subprocess

# Dataset setup imports
from analysis.utils.setup_dataloaders import setup_dataset_HT_H24, setup_dataset_PAVIA_ATN

from careamics.lvae_training.eval_utils import (
    get_predictions,
    stitch_predictions_windowed, 
    stitch_predictions_windowed_from_dir
)
from usplit.core.tiff_reader import save_tiff

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
    Build raw predictions folder under save_dir/raw_predictions_windowed
    """
    raw_dir = save_dir / f"raw_predictions_windowed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def fast_delete(save_dir):
    subprocess.run(["rm", "-rf", str(save_dir)], check=True)

# -----------------------------------------------------------------------------
# Dataset-specific configuration
# -----------------------------------------------------------------------------

def get_dataset_config(dataset_name):
    """
    Returns dataset-specific configuration parameters.
    """
    configs = {
        "PAVIA_ATN": {
            "inner_fraction": 0.5,
            "transpose_dims": (0, 3, 1, 2),  # (N, C, H, W)
            "reshape_dims": (1, 1, 1, -1),
        },
        "HT_H24": {
            "inner_fraction": [1, 0.5, 0.5],  # [Z, Y, X]
            "transpose_dims": (0, 4, 1, 2, 3),  # (N, C, Z, H, W)
            "reshape_dims": (1, 1, 1, 1, -1),
        }
    }
    
    if dataset_name.upper() not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(configs.keys())}")
    
    return configs[dataset_name.upper()]

# -----------------------------------------------------------------------------
# Inference functions
# -----------------------------------------------------------------------------

def run_inference_original(
    model,
    train_dset,
    test_dset,
    batch_size=128,
    num_workers=6,
    mmse_count=64,
    grid_size=32,
    results_root="./results",
    dataset="PAVIA_ATN",
):
    """
    Run standard (non-sliding window) inference.
    Only works for PAVIA_ATN dataset.
    """
    if dataset.upper() != "PAVIA_ATN":
        raise ValueError("Original inference mode only supported for PAVIA_ATN dataset")
    
    print("🚀 Running inference (non-sliding mode)...")
    stitched_predictions_, stitched_stds_ = get_predictions(
        model=model,
        dset=test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        tile_size=(64, 64),
        grid_size=grid_size,
        sliding_window_flag=False,
    )

    def process_preds(stitched_predictions, stitched_stds, train_dset, key, TARGET_CHANNEL_IDX_LIST=[0, 1]):
        stitched_predictions = stitched_predictions[key][..., : len(TARGET_CHANNEL_IDX_LIST)]
        stitched_stds = stitched_stds[key][..., : len(TARGET_CHANNEL_IDX_LIST)]
        mean_params, std_params = train_dset.get_mean_std()
        unnorm = (
            stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1)
            + mean_params["target"].squeeze().reshape(1, 1, 1, -1)
        )
        return unnorm

    stitched_predictions = process_preds(stitched_predictions_, stitched_stds_, train_dset, key=test_dset._fpath.name)
    save_dir = get_save_dir(results_root, dataset=dataset)
    save_tiff(save_dir / "pred_test_dset_microsplit_og.tiff", stitched_predictions.transpose(0, 3, 1, 2))
    with open(save_dir / "pred_test_dset_microsplit_og.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)
    print(f"✅ Saved predictions to {save_dir}")

def run_inference_sliding(
    model, 
    test_dset, 
    results_root,
    batch_size=32, 
    num_workers=4, 
    dataset="PAVIA_ATN"
):
    """
    Run model inference on the test dataset, predict all tiles, and save them as .npy files.
    Yields predictions for stitching.
    
    Args:
        model: PyTorch model
        test_dset: Dataset for inference
        results_root: Root directory to save results
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        dataset: Dataset name (for directory structure)
    """
    print("Initialising sliding window inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    
    save_dir = get_save_dir(results_root, dataset)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Dataloader")
    dloader = DataLoader(test_dset, pin_memory=True, num_workers=num_workers,
                        shuffle=False, batch_size=batch_size)
    
    print("Predicting now....")
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp = batch[0].to(device)
            rec, _ = model(inp)
            
            # get reconstructed img
            if model.model.predict_logvar is None:
                rec_img = rec
                logvar = torch.tensor([-1])
            else:
                rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
            
            rec_np = rec_img.cpu().numpy()  # shape: (batch_size, C, H, W) or (batch_size, C, Z, H, W)
            
            for i in range(rec_np.shape[0]):
                raw_pred_path = get_raw_preds_dir(save_dir)
                pred_path = raw_pred_path / f"pred_{global_idx:010d}.npy"
                np.save(pred_path, rec_np[i])
                global_idx += 1
                yield rec_np[i]
    
    print(f"✅ Saved {global_idx} prediction tiles to {save_dir}")

def run_inference_sliding_and_stitch(
    model,
    train_dset,
    test_dset,
    batch_size=128,
    num_workers=4,
    results_root="./results",
    dataset="PAVIA_ATN",
):
    """
    Run sliding-window inference and stitch predictions.
    Works for both PAVIA_ATN and HT_H24 datasets.
    """
    # Get dataset-specific configuration
    config = get_dataset_config(dataset)
    
    prediction_generator = run_inference_sliding(
        model,
        test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        results_root=results_root,
        dataset=dataset,
    )

    save_dir = get_save_dir(results_root, dataset)
    
    print(f"Stitching predictions with inner_fraction={config['inner_fraction']}...")
    stitched_predictions, coverage_mask = stitch_predictions_windowed(
        prediction_generator,
        test_dset,
        len(test_dset),
        inner_fraction=config['inner_fraction']
    )
    
    # Unnormalize predictions
    mean_params, std_params = train_dset.get_mean_std()
    reshape_dims = config['reshape_dims']
    
    stitched_predictions = (
        stitched_predictions * std_params["target"].squeeze().reshape(*reshape_dims)
        + mean_params["target"].squeeze().reshape(*reshape_dims)
    )
    
    # Save with dataset-specific transpose
    save_tiff(
        save_dir / "pred_test_dset_microsplit_stitched.tiff", 
        stitched_predictions.transpose(*config['transpose_dims'])
    )
    with open(save_dir / "pred_test_dset_microsplit_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)
    
    print(f"✅ Saved stitched predictions to {save_dir}")
    return stitched_predictions

def stitch_predictions_from_dir_only(
    train_dset,
    test_dset,
    results_root,
    dataset="PAVIA_ATN",
    pred_dir_name=None,
    use_memmap=True,
    digits=10,
    batch_size=64,
):
    """
    Stitch raw predictions from directory into a final image, without running inference.
    Works for both PAVIA_ATN and HT_H24 datasets.
    """
    # Get dataset-specific configuration
    config = get_dataset_config(dataset)
    
    save_dir = get_save_dir(results_root, dataset)
    
    if pred_dir_name is None:
        raw_preds_dir = get_raw_preds_dir(save_dir)
    else:
        raw_preds_dir = Path(pred_dir_name)
    
    raw_preds_dir.mkdir(exist_ok=True, parents=True)
    print(f"📂 Reading raw predictions from: {raw_preds_dir}")
    
    stitched_predictions, counts = stitch_predictions_windowed_from_dir(
        pred_dir=raw_preds_dir,
        dset=test_dset,
        num_workers=6,
        inner_fraction=config['inner_fraction'],
        num_patches=len(test_dset),
        batch_size=batch_size * 5,
        digits=digits,
        use_memmap=use_memmap,
        debug=False,
    )
    
    # Unnormalize predictions
    mean_params, std_params = train_dset.get_mean_std()
    reshape_dims = config['reshape_dims']
    
    stitched_predictions = (
        stitched_predictions * std_params["target"].squeeze().reshape(*reshape_dims)
        + mean_params["target"].squeeze().reshape(*reshape_dims)
    )
    
    # Save with dataset-specific transpose
    save_tiff(
        save_dir / "pred_test_dset_microsplit_stitched.tiff", 
        stitched_predictions.transpose(*config['transpose_dims'])
    )
    with open(save_dir / "pred_test_dset_microsplit_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)
    
    print(f"✅ Saved stitched predictions to {save_dir}")
    return stitched_predictions

# -----------------------------------------------------------------------------
# Dataset Setup
# -----------------------------------------------------------------------------

def setup_dataset(dataset_name, sliding_window_flag=False):
    """
    Setup dataset and model based on dataset name.
    
    Returns:
        model, train_dset, test_dset
    """
    dataset_name = dataset_name.upper()
    
    if dataset_name == "PAVIA_ATN":
        print(f"🔧 Setting up {dataset_name} dataset...")
        model, config, (train_dset, val_dset, test_dset), ckpt_dir = setup_environment_PAVIA_ATN(
            sliding_window_flag=sliding_window_flag,
        )
        return model, train_dset, test_dset
        
    elif dataset_name == "HT_H24":
        print(f"🔧 Setting up {dataset_name} dataset...")

        
        model, _, train_dset, _, test_dset = setup_dataset_HT_H24(sliding_window_flag=sliding_window_flag)
    
        
        return model, train_dset, test_dset
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: PAVIA_ATN, HT_H24")

# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on multiple datasets")
    
    # Dataset selection
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["PAVIA_ATN", "HT_H24"],
        help="Dataset to run inference on"
    )
    
    # Inference mode
    parser.add_argument(
        "--sliding_window_flag", 
        action="store_true", 
        help="Enable sliding-window inference"
    )
    parser.add_argument(
        "--stitch_only", 
        action="store_true", 
        help="Stitch previously saved tiles only (skip inference)"
    )
    parser.add_argument(
        "--original_mode",
        action="store_true",
        help="Use original non-sliding inference (PAVIA_ATN only)"
    )
    
    # Paths
    parser.add_argument(
        "--results_root", 
        type=str, 
        default="./Microsplit_predictions", 
        help="Root directory to save results"
    )
    parser.add_argument(
        "--raw_preds_dir", 
        type=str, 
        default=None, 
        help="Directory with raw .npy tiles (for stitch_only mode)"
    )
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    
    # Original mode parameters (PAVIA_ATN only)
    parser.add_argument("--mmse_count", type=int, default=64, help="MMSE count for original mode")
    parser.add_argument("--grid_size", type=int, default=32, help="Grid size for original mode")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.original_mode and args.dataset.upper() != "PAVIA_ATN":
        raise ValueError("Original mode (--original_mode) is only supported for PAVIA_ATN dataset")
    
    if args.original_mode and args.sliding_window_flag:
        raise ValueError("Cannot use both --original_mode and --sliding_window_flag")
    
    # -------------------------------------------------------------------------
    # Setup environment
    # -------------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print(f"🚀 Running inference for {args.dataset.upper()} dataset")
    print(f"{'='*70}\n")
    
    model, train_dset, test_dset = setup_dataset(
        args.dataset, 
        sliding_window_flag=args.sliding_window_flag
    )
    
    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    
    if args.stitch_only:
        print("\n🧵 Stitching only mode...")
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            results_root=args.results_root,
            dataset=args.dataset.upper(),
            pred_dir_name=args.raw_preds_dir,
            batch_size=args.batch_size,
        )
    
    elif args.original_mode:
        print("\n🎯 Running standard (original) inference mode...")
        run_inference_original(
            model,
            train_dset,
            test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mmse_count=args.mmse_count,
            grid_size=args.grid_size,
            results_root=args.results_root,
            dataset=args.dataset.upper(),
        )
    
    elif args.sliding_window_flag:
        print("\n🪟 Running sliding-window inference + stitching...")
        run_inference_sliding_and_stitch(
            model,
            train_dset,
            test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root,
            dataset=args.dataset.upper(),
        )
    
    else:
        raise ValueError(
            "Please specify inference mode: --sliding_window_flag, --original_mode, or --stitch_only"
        )
    
    print(f"\n{'='*70}")
    print(f"✅ Inference complete for {args.dataset.upper()}!")
    print(f"{'='*70}\n")
