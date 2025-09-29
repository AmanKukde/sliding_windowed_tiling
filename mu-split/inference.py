# %%
import os
from pathlib import Path
import glob
import dill
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from train_setup import setup_environment, get_best_checkpoint  # your fixed setup
from careamics.lvae_training.eval_utils import get_device, stitch_and_crop_predictions_inner_tile, get_predictions
from usplit.core.tiff_reader import save_tiff
from usplit.analysis.lvae_utils import get_img_from_forward_output


# %%
def get_save_dir(results_root, dataset, ckpt_dir):
    """
    Build save path as results_root/dataset/modality/LCtype
    """
    parts = Path(ckpt_dir).parts
    modality, lctype = parts[-2], parts[-1]
    save_dir = Path(results_root) / dataset / modality / lctype
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_raw_preds_dir(save_dir, dataset, ckpt_dir):
    """
    Build raw predictions folder as save_dir/raw_predictions_{dataset}_{modality}_{lctype}_predid
    """
    parts = Path(ckpt_dir).parts
    modality, lctype = parts[-2], parts[-1]
    raw_dir = save_dir / f"raw_predictions_{dataset}_{modality}_{lctype}_predid"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


# %%
def run_inference_sliding(model, train_dset, test_dset, dataset, ckpt_dir,
                          batch_size=32, num_workers=6,
                          results_root="/group/jug/aman/ConsolidatedResults/Results_usplit_64"):

    device = get_device()
    dloader = DataLoader(test_dset, pin_memory=True, num_workers=num_workers,
                         shuffle=False, batch_size=batch_size*15)

    model.eval().to(device)

    # Raw predictions folder
    save_dir = get_save_dir(results_root, dataset, ckpt_dir)
    raw_preds_dir = get_raw_preds_dir(save_dir, dataset, ckpt_dir)

    # Determine which tiles are already saved
    total_tiles = len(test_dset)
    existing_preds = set(f.name for f in raw_preds_dir.glob("pred_*.npy"))
    missing_indices = [i for i in range(total_tiles) if f"pred_{i:06d}.npy" not in existing_preds]

    if len(missing_indices) == 0:
        print("All predictions already exist. Skipping model inference.")
    else:
        print(f"Predicting {len(missing_indices)} missing tiles out of {total_tiles}")

        # Save predictions one by one
        with torch.no_grad():
            global_idx = 0
            for batch in tqdm(dloader, desc="Predicting tiles"):
                start_idx = global_idx
                end_idx = global_idx + batch[0].shape[0]
                batch_indices = range(start_idx, end_idx)
                # Determine which indices in this batch are missing
                batch_missing_mask = [idx in missing_indices for idx in batch_indices]
                if not any(batch_missing_mask):
                    global_idx += batch[0].shape[0]
                    continue  # all tiles in this batch already exist

                # Only compute predictions for missing tiles
                inp, tar = batch
                inp = inp.to(device)
                rec, _ = model(inp)
                rec = get_img_from_forward_output(rec, model)
                rec_np = rec.cpu().numpy()

                for i, missing in enumerate(batch_missing_mask):
                    if not missing:
                        global_idx += 1
                        continue
                    pred_path = raw_preds_dir / f"pred_{global_idx:06d}.npy"
                    np.save(pred_path, rec_np[i])
                    global_idx += 1

    # Load all saved predictions
    pred_files = sorted(raw_preds_dir.glob("pred_*.npy"))
    tiles_arr = np.stack([np.load(f) for f in pred_files], axis=0)
    print(f"Loaded {len(tiles_arr)} predictions from {raw_preds_dir}")

    # Stitch
    stitched_predictions, _ = stitch_and_crop_predictions_inner_tile(tiles_arr, test_dset)

    # Unnormalize
    mean_params, std_params = train_dset.get_mean_std()
    stitched_predictions = stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1) + \
                           mean_params["target"].squeeze().reshape(1, 1, 1, -1)

    # Save final stitched output
    save_tiff(save_dir / "pred_test_dset_usplit_windowed.tiff", stitched_predictions.transpose(0,3,1,2))
    with open(save_dir / "pred_test_dset_usplit_windowed.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"Saved sliding-window predictions to {save_dir}")

# %%
def run_inference_original(model, train_dset, test_dset, dataset, ckpt_dir,
                           batch_size=128*5, num_workers=6, mmse_count=64, grid_size=32,
                           results_root="/group/jug/aman/ConsolidatedResults/Results_usplit_64"):

    stitched_predictions_, stitched_stds_ = get_predictions(
        model=model,
        dset=test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        tile_size=(64, 64),
        grid_size=grid_size,
        sliding_window_flag=False
    )

    # Process predictions
    def process_preds(stitched_predictions, stitched_stds, dset, key, TARGET_CHANNEL_IDX_LIST=[0, 1]):
        stitched_predictions = stitched_predictions[key][..., :len(TARGET_CHANNEL_IDX_LIST)]
        stitched_stds = stitched_stds[key][..., :len(TARGET_CHANNEL_IDX_LIST)]
        mean_params, std_params = dset.get_mean_std()
        unnorm = stitched_predictions * std_params["target"].squeeze().reshape(1,1,1,-1) + \
                 mean_params["target"].squeeze().reshape(1,1,1,-1)
        return unnorm, stitched_predictions, stitched_stds

    stitched_predictions, norm_preds, stitched_stds = process_preds(
        stitched_predictions_, stitched_stds_, train_dset, dataset
    )

    # Save
    save_dir = get_save_dir(results_root, dataset, ckpt_dir)
    save_tiff(save_dir / "pred_test_dset_usplit_og.tiff", stitched_predictions.transpose(0,3,1,2))
    with open(save_dir / "pred_test_dset_usplit_og.pkl", "wb") as f:
        dill.dump(stitched_predictions, f)

    print(f"Saved original predictions to {save_dir}")


# %%
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LVAE inference")

    # Required sweep args
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. Hagen, PAVIA_ATN)")
    parser.add_argument("--modality", type=str, required=True,
                        help="Data modality (e.g. MitoVsAct, ActTub)")
    parser.add_argument("--lc_type", type=str, required=True,
                        help="LC type (e.g. LeanLC, DeepLC)")
    parser.add_argument("--sliding_window_flag", action="store_true",
                        help="Enable sliding-window inference (default: off)")

    # Optional overrides
    parser.add_argument("--results_root", type=str,
                        default="/group/jug/aman/ConsolidatedResults/Results_usplit_64",
                        help="Where to save results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (used for sliding-window mode)")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of data loader workers")
    parser.add_argument("--mmse_count", type=int, default=64,
                        help="MMSE count (only for original mode)")
    parser.add_argument("--grid_size", type=int, default=32,
                        help="Grid size (only for original mode)")

    args = parser.parse_args()

    # ------------------------
    # Setup
    # ------------------------
    model, config, (train_dset, val_dset, test_dset), ckpt_dir = setup_environment(
        dataset_name=args.dataset,
        lc_type=args.lc_type,
        modality=args.modality,
        img_sz=64,
        sliding_window_flag=args.sliding_window_flag
    )

    # ------------------------
    # Run inference
    # ------------------------
    if args.sliding_window_flag:
        run_inference_sliding(
            model, train_dset, test_dset, args.dataset, ckpt_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            results_root=args.results_root
        )
    else:
        run_inference_original(
            model, train_dset, test_dset, args.dataset, ckpt_dir,
            batch_size=args.batch_size * 5,
            num_workers=args.num_workers,
            mmse_count=args.mmse_count,
            grid_size=args.grid_size,
            results_root=args.results_root
        )
