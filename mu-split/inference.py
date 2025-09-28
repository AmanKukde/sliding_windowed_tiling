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

    # Save predictions one by one
    with torch.no_grad():
        global_idx = 0
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp, tar = batch
            inp = inp.to(device)
            rec, _ = model(inp)
            rec = get_img_from_forward_output(rec, model)
            rec_np = rec.cpu().numpy()

            # Save each prediction individually
            for i in range(rec_np.shape[0]):
                pred_path = raw_preds_dir / f"pred_{global_idx:06d}.npy"
                if pred_path.exists():
                    global_idx += 1
                    continue  # resume logic
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
if __name__ == "__main__":
    # Setup
    dataset_name = "Hagen"
    lc_type = "LeanLC"
    modality = "MitoVsAct"
    sliding_window_flag = True  # change to False if you want original predictions

    model, config, (train_dset, val_dset, test_dset), ckpt_dir = setup_environment(
        dataset_name=dataset_name,
        lc_type=lc_type,
        modality=modality,
        img_sz=64,
        sliding_window_flag=sliding_window_flag
    )

    # Run inference
    if sliding_window_flag:
        run_inference_sliding(model, train_dset, test_dset, dataset_name, ckpt_dir)
    else:
        run_inference_original(model, train_dset, test_dset, dataset_name, ckpt_dir)
