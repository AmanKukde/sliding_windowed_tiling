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
import os
import sys
sys.path.append(os.path.expanduser('~/sliding_windowed_tiling/'))
sys.path.append(os.path.expanduser('~/sliding_windowed_tiling/analysis/'))

from setup_dataloaders import setup_dataset_PAVIA_ATN
from careamics.lvae_training.eval_utils import (
    stitch_predictions_windowed_from_dir,
    get_predictions,
    stitch_predictions_windowed,
    stitch_predictions_windowed_highperf
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
    batch_size=128,
    num_workers=6,
    mmse_count=64,
    grid_size=32,
    results_root="./results_PAVIA_ATN",
):  
    from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    get_unnormalized_predictions)
    print("🚀 Running inference (non-sliding mode)...")
    stitched_predictions, norm_stitched_predictions, stitched_stds = (
    get_unnormalized_predictions(
        model,
        test_dset,
        data_key=test_dset._fpath.name,
        mmse_count=mmse_count,
        grid_size=32,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    )

    save_dir = get_save_dir(results_root, dataset="PAVIA_ATN")
    save_tiff(save_dir / "pred_test_dset_microsplit_og.tiff", stitched_predictions.transpose(0, 3, 1, 2))
    with open(save_dir / "pred_test_dset_microsplit_og.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)
    print(f"✅ Saved predictions to {save_dir}")


def run_inference_sliding(model, test_dset, results_root,
                        batch_size=32, num_workers=4, dataset="PAVIA_ATN"):
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
                yield rec_np[i]             
    print(f"✅ Saved {global_idx} prediction tiles to {save_dir}")

def run_inference_sliding_and_stitch(
    model,
    train_dset,
    test_dset,
    batch_size=128,
    num_workers=4,
    results_root="./results_PAVIA_ATN",
    dataset="PAVIA_ATN",
):
    """
    Run sliding-window inference and stitch predictions.
    """
    prediction_generator = run_inference_sliding(
        model,
        test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        results_root=results_root,
        dataset=dataset,
    )
    save_dir = get_save_dir(results_root,dataset)

    stitched_predictions, coverage_mask = stitch_predictions_windowed(prediction_generator,test_dset,len(test_dset),inner_fraction=0.5, debug=True)
    #unnormalizing the predictions
    mean_params, std_params = train_dset.get_mean_std() #!@VERA which dataset to use here to Unnormalise ? does the values from train get passed to test?
    stitched_predictions = (
        stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1)
        + mean_params["target"].squeeze().reshape(1, 1, 1, -1)
    )

    save_tiff(save_dir / "pred_test_dset_microsplit_stitched.tiff", stitched_predictions.transpose(0, 3, 1, 2))
    with open(save_dir / "pred_test_dset_microsplit_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"✅ Saved stitched predictions to {save_dir}")

def stitch_predictions_from_dir_only(
    train_dset,
    test_dset,
    results_root, 
    dataset = "PAVIA_ATN",
    pred_dir_name=None,
    use_memmap=True,
    digits=10,
    batch_size=64,
    channels = 2,
):
    """
    Stitch raw predictions into a final image, without inference.
    """
    save_dir = get_save_dir(results_root,dataset)
    if pred_dir_name is None:
        raw_preds_dir = get_raw_preds_dir(save_dir)
    else:
        raw_preds_dir = Path(pred_dir_name)
    raw_preds_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 Reading raw predictions from: {raw_preds_dir}")

    # stitched_predictions, counts = stitch_predictions_windowed_from_dir(
    #     pred_dir=raw_preds_dir,
    #     dset=test_dset,
    #     num_workers=6,
    #     inner_fraction=0.5,
    #     num_patches = len(test_dset),
    #     batch_size=batch_size * 5,
    #     digits=digits,
    #     debug=True,
    # )
    stitched_predictions, counts = stitch_predictions_windowed_highperf(
        pred_dir=raw_preds_dir,
        dset=test_dset,
        num_workers=6,
        inner_fraction=0.5,
        num_patches = len(test_dset),
        batch_size=batch_size * 500,
        digits=digits,
        debug=True,
    )

    mean_params, std_params = train_dset.get_mean_std()
    stitched_predictions = (
        stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1)
        + mean_params["target"].squeeze().reshape(1, 1, 1, -1)
    )

    save_tiff(save_dir / "pred_test_dset_microsplit_stitched.tiff", stitched_predictions.transpose(0, 3, 1, 2))
    with open(save_dir / "pred_test_dset_microsplit_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"✅ Saved stitched predictions to {save_dir}")
    return stitched_predictions
# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PAVIA_ATN LVAE inference")
    parser.add_argument("--num_channels", type=int, default=2, help="Number of channels (2, 3, or 4)")
    parser.add_argument("--reduce_dataset_size", action="store_true", help="Reduce dataset size for quick testing but only for val_dset or test_dset based on what you selected earlier") #!Reduce Dataset Size is set as True by default
    parser.add_argument("--sliding_window_flag", action="store_true", help="Enable sliding-window inference")
    parser.add_argument("--stitch_only", action="store_true", help="Stitch previously saved tiles only")
    parser.add_argument("--results_root", type=str, default="./Microsplit_predictions", help="Where to save results")
    parser.add_argument("--raw_preds_dir", type=str, default=None, help="Directory with raw .npy tiles")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mmse_count", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--load_pretrained_ckpt", action="store_false", help="Skip pretrained model loading")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup environment
    # -------------------------------------------------------------------------

    model, experiment_config, train_dset, val_dset, test_dset = setup_dataset_PAVIA_ATN(
        sliding_window_flag=args.sliding_window_flag,
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    # test_dset.reduce_data([0])  #! For quick testing, remove later 

    if args.stitch_only:
        print("🧵 Stitching only...")
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            results_root=args.results_root,
            batch_size=args.batch_size,
            channels = args.num_channels,
        )
    elif args.sliding_window_flag:
        print("🪟 Running sliding-window inference + stitching")
        run_inference_sliding_and_stitch(
            model,
            train_dset,
            test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root,
        )
    else:
        print("🎯 Running standard inference")
        run_inference_original(
            model,
            train_dset,
            test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mmse_count=args.mmse_count,
            grid_size=args.grid_size,
            results_root=args.results_root,
        )
