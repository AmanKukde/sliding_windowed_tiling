import dill
from pathlib import Path
from careamics.lvae_training.eval_utils import (
    stitch_predictions_windowed, stitch_predictions_windowed_from_dir
)
from usplit.core.tiff_reader import save_tiff


# Dataset specific imports...

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import pooch
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser('~/sliding_windowed_tiling/'))
sys.path.append(os.path.expanduser('~/sliding_windowed_tiling/analysis/'))
from analysis.utils.setup_dataloaders import setup_dataset_HT_H24



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

def run_inference_sliding(model, test_dset, results_root,
                        batch_size=32, num_workers=4, dataset="HT_H24"):
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
                yield rec_np[i]                
    print(f"✅ Saved {global_idx} prediction tiles to {save_dir}")

def run_inference_sliding_and_stitch(
    model,
    train_dset,
    test_dset,
    batch_size=128,
    num_workers=4,
    results_root="./results_HT_H24",
    dataset="HT_H24",
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

    stitched_predictions, coverage_mask = stitch_predictions_windowed(prediction_generator,test_dset,len(test_dset),inner_fraction= [1,0.5,0.5])

    mean_params, std_params = train_dset.get_mean_std()
    stitched_predictions = stitched_predictions * std_params[
        "target"
    ].squeeze().reshape(1, 1, 1, 1, -1) + mean_params["target"].squeeze().reshape(
        1, 1, 1, 1, -1
    )

    save_tiff(save_dir / "pred_test_dset_microsplit_stitched.tiff", stitched_predictions.transpose(0, 4, 1, 2, 3))
    with open(save_dir / "pred_test_dset_microsplit_stitched.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"✅ Saved stitched predictions to {save_dir}")
    return stitched_predictions

def stitch_predictions_from_dir_only(
    train_dset,
    test_dset,
    results_root, 
    dataset = "HT_LIF24",
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

    stitched_predictions, counts = stitch_predictions_windowed_from_dir(
        pred_dir=raw_preds_dir,
        dset=test_dset,
        num_workers=6,
        inner_fraction=0.5,
        num_patches = len(test_dset),
        batch_size=batch_size * 5,
        digits=digits,
        use_memmap=use_memmap,
        debug=False,
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run sliding-window inference and stitching for HT_H24 dataset")
    parser.add_argument("--results_root",type=str,default="./results_HT_H24",help="Root directory to save results",)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--sliding_window_flag", action="store_true", help="Whether to use sliding window datasets")
    parser.add_argument("--stitch_only", action="store_true", help="If set, only stitch from existing raw predictions without running inference")
    args = parser.parse_args()

    model, _, train_dset, _, test_dset = setup_dataset_HT_H24(sliding_window_flag=args.sliding_window_flag)
    if args.stitch_only:
        print("🧩 Stitching only from existing raw predictions")
        stitch_predictions_from_dir_only(
            train_dset,
            test_dset,
            results_root=args.results_root,
            dataset="HT_H24",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        print("🪟 Running sliding-window inference + stitching")
        run_inference_sliding_and_stitch(
            model,
            train_dset,
            test_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root,
            dataset="HT_H24",
        )