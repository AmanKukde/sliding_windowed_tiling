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
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import get_microsplit_parameters
from microsplit_reproducibility.datasets import create_train_val_datasets
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


def setup_dataset(dataset_name, img_sz=64, sliding_window_flag=False):
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


def compute_and_save_stats(tar, img_og, img_sw, save_dir, dataset_name):
    stats_og, stats_win = {}, {}
    for i in tqdm(range(len(img_og)), desc="Computing high-SNR stats"):
        stats_og[i] = compute_high_snr_stats(tar[i:i+1], img_og[i:i+1])
        stats_win[i] = compute_high_snr_stats(tar[i:i+1], img_sw[i:i+1])

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"stats_og_{dataset_name}.pkl", "wb") as f:
        dill.dump(stats_og, f)
    with open(save_dir / f"stats_win_{dataset_name}.pkl", "wb") as f:
        dill.dump(stats_win, f)

    return stats_og, stats_win


# ------------------------------
# Analyses
# ------------------------------
def run_gradient_based_analysis(img_sw, img_og, save_dir, tile_size=32, bins=2000, channel=1, kl_start=29, kl_end=33):
    grad_sw = GradientUtils(img_sw, tile_size=tile_size)
    grad_og = GradientUtils(img_og, tile_size=tile_size)
    bin_edges = grad_sw.make_bin_edges(n_bins=bins)

    summarize_gradients(
        grad_utils_og=grad_og,
        grad_utils_sw=grad_sw,
        bin_edges=bin_edges,
        channel=channel,
        save_dir=save_dir,
    )


def run_qualitative_analysis(test_dset, img_sw, img_og, save_dir):
    tar = get_target(test_dset)
    inp = get_input(test_dset)

    # --- PSNR analysis ---
    psnr_df, avg_psnr_df = compute_psnr_and_plot(tar, img_og, img_sw, save_dir)
    plot_psnr_difference(avg_psnr_df,save_dir=save_dir)
    plot_avg_psnr_zoomed(avg_psnr_df,save_dir=save_dir)
    plot_all_psnr(avg_psnr_df,save_dir = save_dir)

    # --- High-SNR stats ---
    stats_og, stats_win = compute_and_save_stats(tar, img_og, img_sw, save_dir, "dataset")

    # --- Full-frame evaluation plots ---
    frame_dir = save_dir / "Frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(test_dset._data)):
        full_frame_evaluation(
            predictions_list=[img_sw, img_og],
            tar_list=[test_dset._data, test_dset._data],
            inp_list=[inp, inp],
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
    parser.add_argument("--dataset", required=True, choices=["Hagen", "PAVIA_ATN"])
    parser.add_argument("--pred_sw", required=True, help="Sliding window predictions (.tiff/.pkl)")
    parser.add_argument("--pred_og", required=True, help="Original predictions (.tiff/.pkl)")
    parser.add_argument("--save_dir", required=True, help="Directory to save plots and stats")
    parser.add_argument("--tile_size", type=int, default=32)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--kl_start", type=int, default=29)
    parser.add_argument("--kl_end", type=int, default=33)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--gradient_based_analysis", action="store_true")
    parser.add_argument("--qualitative_analysis", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all analyses if no flags specified")

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

    if run_qualitative:
        print("Loading full dataset (may take time)...")
        test_dset = setup_dataset(args.dataset)
        print("Running qualitative analysis (PSNR, metrics, frame plots)...")
        run_qualitative_analysis(test_dset, img_sw, img_og, save_dir)

    if run_gradient:
        print("Running gradient-based analysis...")
        run_gradient_based_analysis(
            img_sw, img_og, save_dir,
            tile_size=args.tile_size,
            bins=args.bins,
            channel=args.channel,
            kl_start=args.kl_start,
            kl_end=args.kl_end
        )




if __name__ == "__main__":
    main()
