#!/usr/bin/env python3
# analyze_experiment.py

import argparse
from pathlib import Path
import pickle
import dill
import tifffile as tiff
import numpy as np
from tqdm import tqdm
import pandas as pd

# Project-specific imports
from utils.plot_utils import (
    full_frame_evaluation,
    compute_psnr_and_plot,
    plot_psnr_difference,
    plot_avg_psnr_zoomed,
    plot_all_psnr
)
from utils.gradient_utils import GradientUtils
from utils.analysis_utils import summarize_gradients
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import get_input, get_target

from microsplit_reproducibility.utils.paper_metrics import compute_high_snr_stats

try:
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass


# ------------------------------
# Utilities
# ------------------------------
def load_prediction(path):
    path = Path(path)
    if path.suffix in [".pkl", ".dill"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        # TIFF expected shape: (N, C, H, W)
        return tiff.imread(path).transpose(0, 2, 3, 1)


def compute_and_save_stats(tar, img_og, img_sw, save_dir, dataset_name):
    stats_og, stats_win = {}, {}

    for i in tqdm(range(len(img_og)), desc="Computing high-SNR stats"):
        stats_og[i] = compute_high_snr_stats(tar[i:i+1], img_og[i:i+1])
        stats_win[i] = compute_high_snr_stats(tar[i:i+1], img_sw[i:i+1])

    # ensure directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # save as pickle
    with open(save_dir / f"stats_og_{dataset_name}.pkl", "wb") as f:
        dill.dump(stats_og, f)
    with open(save_dir / f"stats_win_{dataset_name}.pkl", "wb") as f:
        dill.dump(stats_win, f)

    # convert to DataFrames
    df_og = pd.DataFrame.from_dict(stats_og, orient="index")
    df_win = pd.DataFrame.from_dict(stats_win, orient="index")

    # save as Excel
    df_og.to_csv(save_dir / f"stats_og_{dataset_name}.csv", index=True)
    df_win.to_csv(save_dir / f"stats_win_{dataset_name}.csv", index=True)


    return stats_og, stats_win

# ------------------------------
# Analyses
# ------------------------------
def run_gradient_based_analysis(img_sw, img_og, save_dir, inner_tile_size=32, bins=200, channel=1, kl_start=29, kl_end=33):
    grad_sw = GradientUtils(img_sw, tile_size=inner_tile_size)
    grad_og = GradientUtils(img_og, tile_size=inner_tile_size)
    bin_edges = grad_sw.make_bin_edges(n_bins=bins)

    summarize_gradients(
        grad_utils_og=grad_og,
        grad_utils_sw=grad_sw,
        bin_edges=bin_edges,
        channel=channel,
        save_dir= save_dir / f"Gradient_Analysis_Channel_{channel}")



def run_qualitative_analysis(test_dset, img_sw, img_og, save_dir, dataset_name):
    tar = get_target(test_dset)
    inp = get_input(test_dset)
        # --- High-SNR stats ---

    if hasattr(test_dset, "_tar_idx_list") and test_dset._tar_idx_list:
        tar = tar[..., test_dset._tar_idx_list]
        inp = inp[..., test_dset._tar_idx_list]
    stats_og, stats_win = compute_and_save_stats(tar, img_og, img_sw, save_dir, dataset_name)

    if tar.shape.n_dims == 4:
        # # --- PSNR analysis ---
        psnr_df, avg_psnr_df = compute_psnr_and_plot(tar, img_og, img_sw, save_dir)

        plot_psnr_difference(avg_psnr_df,save_dir=save_dir)
        plot_avg_psnr_zoomed(avg_psnr_df,save_dir=save_dir)
        plot_all_psnr(avg_psnr_df,save_dir = save_dir)

        # --- Full-frame evaluation plots ---
        frame_dir = save_dir / "Frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(len(test_dset._data)):
            full_frame_evaluation(
                predictions_list=[img_sw[idx], img_og[idx]],
                tar_list=[test_dset._data[idx], test_dset._data[idx]],
                inp_list=[inp[idx], inp[idx]],
                metrics_list=[stats_win[idx], stats_og[idx]],
                frame_idx=idx,
                titles=["Sliding Window", "Original"],
                save_path=frame_dir / f"{idx+1}.png"
            )
# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full experiment analysis")
    parser.add_argument("--model_name", required=True, choices=["usplit", "microsplit"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_sw", required=True, help="Sliding window predictions (.tiff/.pkl)")
    parser.add_argument("--pred_og", required=True, help="Original predictions (.tiff/.pkl)")
    parser.add_argument("--save_dir", required=True, help="Directory to save plots and stats")
    parser.add_argument("--inner_tile_size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--kl_start", type=int, default=29)
    parser.add_argument("--kl_end", type=int, default=33)
    parser.add_argument("--channel", default="all")
    parser.add_argument("--gradient_based_analysis", type=bool, default=True)
    parser.add_argument("--qualitative_analysis", type=bool, default=True)
    parser.add_argument("--all", type = bool, default = False, help="Run all analyses if no flags specified")

    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    img_sw = load_prediction(args.pred_sw)
    img_og = load_prediction(args.pred_og)

    # Determine analyses to run
    run_gradient = args.gradient_based_analysis or args.all
    run_qualitative = args.qualitative_analysis or args.all

    # Default: run both if no flags specified
    if not (args.gradient_based_analysis or args.qualitative_analysis or args.all):
        run_gradient = run_qualitative = True
    if run_gradient:
        print("Running gradient-based analysis...")
        if args.channel == "all":
            for channel in [0,1]:
                run_gradient_based_analysis(
                    img_sw, img_og, save_dir,
                    inner_tile_size=args.inner_tile_size,
                    bins=args.bins,
                    channel=channel,
                    kl_start=args.kl_start,
                    kl_end=args.kl_end
                )
        else:
            run_gradient_based_analysis(
                img_sw, img_og, save_dir,
                inner_tile_size=args.tile_size,
                bins=args.bins,
                channel=args.channel,
                kl_start=args.kl_start,
                kl_end=args.kl_end
            )
    if run_qualitative:
        if args.model_name == "usplit":
            from utils.setup_dataloaders import setup_dataset_usplit as setup_dataset
        elif args.model_name == "microsplit ht_lif24":
            from utils.setup_dataloaders import setup_dataset_microsplit_HT_LIF24 as setup_dataset
        elif args.model_name == "microsplit":
            from utils.setup_dataloaders import setup_dataset_microsplit_HT_H24 as setup_dataset
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")
        test_dset = setup_dataset()
        print("Running qualitative analysis (PSNR, metrics, frame plots)...")
        run_qualitative_analysis(test_dset, img_sw, img_og, save_dir, args.dataset)

if __name__ == "__main__":
    main()


