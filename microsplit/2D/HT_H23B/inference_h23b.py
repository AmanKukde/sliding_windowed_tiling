
import os
from pathlib import Path
import re
import dill
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import subprocess

from train_setup_h23b import setup_environment_HT_H23B

from careamics.lvae_training.eval_utils import (
    stitch_and_crop_predictions_inner_tile_from_dir,
    get_predictions,
)

from usplit.core.tiff_reader import save_tiff

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def get_save_dir(results_root, dataset):
    """
    Build save path as results_root/HT_LIF24/
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

# -----------------------------------------------------------------------------
# Inference functions
# -----------------------------------------------------------------------------

def run_inference_original(
    model,
    train_dset,
    test_dset,
    exposure,
    ckpt_dir,
    batch_size=128,
    num_workers=6,
    mmse_count=64,
    grid_size=32,
    results_root="./results_HT_H23B",
):
    print("🚀 Running inference (non-sliding mode)...")

    # test_dset.reduce_data([0]) #! For quick testing, remove later 

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

    def process_preds(stitched_predictions, stitched_stds, dset, key, TARGET_CHANNEL_IDX_LIST=[0, 1]):
        stitched_predictions = np.array(stitched_predictions[key])[..., : len(TARGET_CHANNEL_IDX_LIST)]
        stitched_stds =  np.array(stitched_stds[key])[..., : len(TARGET_CHANNEL_IDX_LIST)]
        mean_params, std_params = dset.get_mean_std()
        unnorm = (
            stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1)
            + mean_params["target"].squeeze().reshape(1, 1, 1, -1)
        )
        return unnorm

    stitched_predictions = process_preds(stitched_predictions_, stitched_stds_, train_dset, key = test_dset._fpath.name)

    save_dir = get_save_dir(results_root, ckpt_dir)
    save_tiff(save_dir / "pred_test_dset_usplit_og.tiff", stitched_predictions.transpose(0, 3, 1, 2))
    with open(save_dir / "pred_test_dset_usplit_og.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"✅ Saved predictions to {save_dir}")

def fast_delete(save_dir):
    subprocess.run(["rm", "-rf", str(save_dir)], check=True)

def run_inference_sliding(model, test_dset, exposure, ckpt_dir, results_root,
                        batch_size=32, num_workers=4, dataset="HT_H23B"):
    """
    Run model inference on the test dataset, predict all tiles, and save them as .npy files.
    Does NOT check for existing predictions and does NOT stitch.
    
    Args:
        model: PyTorch model
        test_dset: Dataset for inference
        save_dir: Directory to save predictions (will be created if doesn't exist)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
    """
    print("Initialising")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    save_dir = get_save_dir(results_root, dataset)
    save_dir = Path(save_dir)
    # fast_delete(save_dir)  # deletes folder and all files 
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

            rec_np = rec_img.cpu().numpy()  # shape: (batch_size, C, H, W)
            
            for i in range(rec_np.shape[0]):
                raw_pred_path = get_raw_preds_dir(save_dir)
                pred_path =  raw_pred_path / f"pred_{global_idx:010d}.npy"
                np.save(pred_path, rec_np[i])
                global_idx += 1
                
    print(f"✅ Saved {global_idx} prediction tiles to {save_dir}")

def stitch_predictions_from_dir_only(
    train_dset,
    test_dset,
    exposure,
    ckpt_dir,
    results_root="./results_HT_H23B",
    pred_dir_name=None,
    use_memmap=True,
    digits=10,
    batch_size=64,
):
    """
    Stitch raw predictions into a final image, without inference.
    """
    save_dir = get_save_dir(results_root, dataset = "HT_H23B")
    if pred_dir_name is None:
        raw_preds_dir = get_raw_preds_dir(save_dir)
    else:
        raw_preds_dir = Path(pred_dir_name)
    raw_preds_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 Reading raw predictions from: {raw_preds_dir}")

    stitched_predictions, counts = stitch_and_crop_predictions_inner_tile_from_dir(
        pred_dir=raw_preds_dir,
        dset=test_dset,
        num_workers=6,
        inner_fraction=0.5,
        batch_size=batch_size * 5,
        digits=digits,
        use_memmap=use_memmap,
    )

    mean_params, std_params = train_dset.get_mean_std()
    stitched_predictions = (
        stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1)
        + mean_params["target"].squeeze().reshape(1, 1, 1, -1)
    )

    save_tiff(save_dir / "pred_test_dset_stitched.tiff", stitched_predictions.transpose(0, 3, 1, 2))
    with open(save_dir / "pred_test_dset_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"✅ Saved stitched predictions to {save_dir}")
    return stitched_predictions

# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run  LVAE inference")
    parser.add_argument("--exposure", type=str, default="5ms", help="Exposure duration (e.g. 2ms, 5ms, 20ms)")
    parser.add_argument("--num_channels", type=int, default=2, help="Number of channels (2, 3, or 4)")
    parser.add_argument("--reduce_dataset_size", action="store_false", help="Reduce dataset size for quick testing but only for val_dset or test_dset based on what you selected earlier") #!Reduce Dataset Size is set as True by default
    parser.add_argument("--sliding_window_flag", action="store_false", help="Enable sliding-window inference")
    parser.add_argument("--stitch_only", action="store_true", help="Stitch previously saved tiles only")
    parser.add_argument("--results_root", type=str, default="./results_HT_H23B", help="Where to save results")
    parser.add_argument("--raw_preds_dir", type=str, default=None, help="Directory with raw .npy tiles")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mmse_count", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--load_pretrained_ckpt", action="store_false", help="Skip pretrained model loading")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup environment
    # -------------------------------------------------------------------------

    model, config, (train_dset, val_dset, test_dset), ckpt_dir = setup_environment_HT_H23B(
        sliding_window_flag=args.sliding_window_flag,
        pretrained=args.load_pretrained_ckpt
    )
    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    if args.stitch_only:
        print("🧵 Stitching only...")
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            args.exposure,
            ckpt_dir,
            results_root=args.results_root,
            pred_dir_name=args.raw_preds_dir,
            batch_size=args.batch_size,
        )
    elif args.sliding_window_flag:
        print("🪟 Running sliding-window inference + stitching")
        run_inference_sliding(
            model,
            test_dset,
            args.exposure,
            ckpt_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root,
        )
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            args.exposure,
            ckpt_dir,
            results_root=args.results_root,
            pred_dir_name=args.raw_preds_dir,
            batch_size=args.batch_size,
        )
    else:
        print("🎯 Running standard inference")
        run_inference_original(
            model,
            train_dset,
            test_dset,
            args.exposure,
            ckpt_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mmse_count=args.mmse_count,
            grid_size=args.grid_size,
            results_root=args.results_root,
        )