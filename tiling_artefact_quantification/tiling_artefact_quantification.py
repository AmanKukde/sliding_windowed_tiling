#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt

from analysis.gradient_utils import GradientUtils
from analysis.plot_utils import plot_multiple_hist, plot_multiple_bar, plot_kl_heatmaps_for_range


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare gradients between Original and Sliding Window methods."
    )
    parser.add_argument("--sw_path", required=True, help="Path to Sliding Window prediction TIFF file")
    parser.add_argument("--og_path", required=True, help="Path to Original prediction TIFF file")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--tile_size", type=int, default=32, help="Tile size used for gradient analysis")
    parser.add_argument("--bins", type=int, default=2000, help="Number of histogram bins")
    parser.add_argument("--channel", type=int, default=1, help="Channel index to use for gradient extraction")
    parser.add_argument("--kl_start", type=int, default=29, help="Start tile index for KL heatmap")
    parser.add_argument("--kl_end", type=int, default=33, help="End tile index for KL heatmap")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # === Load images ===
    print("Loading TIFF images...")
    imgs_sw = imread(args.sw_path).transpose(0, 2, 3, 1)
    imgs_og = imread(args.og_path).transpose(0, 2, 3, 1)

    # === Create GradientUtils instances ===
    grad_utils_sw = GradientUtils(imgs_sw, tile_size=args.tile_size)
    grad_utils_og = GradientUtils(imgs_og, tile_size=args.tile_size)

    print("Extracting gradients...")
    grad_sw_edge = grad_utils_sw.get_gradients_at("edge", channels=args.channel)
    grad_sw_mid = grad_utils_sw.get_gradients_at("middle", channels=args.channel)
    grad_og_edge = grad_utils_og.get_gradients_at("edge", channels=args.channel)
    grad_og_mid = grad_utils_og.get_gradients_at("middle", channels=args.channel)

    # === Shared bin edges ===
    _, bin_edges = np.histogram(
        np.concatenate([grad_og_mid, grad_sw_mid, grad_og_edge, grad_sw_edge]),
        bins=args.bins
    )

    # Reinitialize GradientUtils with shared bins
    grad_utils_sw = GradientUtils(imgs_sw, tile_size=args.tile_size, bin_edges=bin_edges)
    grad_utils_og = GradientUtils(imgs_og, tile_size=args.tile_size, bin_edges=bin_edges)

    # === Compute Peakiness Scores ===
    print("Computing peakiness scores...")
    data = {
        "Original Method": grad_utils_og.get_peakiness_scores(
            grad_utils_og.histogram_edges, grad_utils_og.histogram_middle
        )[-1],
        "Sliding Window Method": grad_utils_sw.get_peakiness_scores(
            grad_utils_sw.histogram_edges, grad_utils_sw.histogram_middle
        )[-1],
    }

    df = pd.DataFrame(data, index=["Peakiness Score (Histogram Middle vs Edge)"])
    df["Δ (SW - OG)"] = df["Sliding Window Method"] - df["Original Method"]

    print("\nPeakiness Scores (lower is better):")
    print(df.to_string())

    if df["Δ (SW - OG)"].iloc[0] < 0:
        print("\n✅ Sliding Window Method performs better (lower Peakiness Score).")

    # Save DataFrame
    csv_path = os.path.join(args.save_dir, "peakiness_scores.csv")
    df.to_csv(csv_path)
    print(f"\nSaved peakiness scores to: {csv_path}")

    # === Plot histograms (raw gradients) ===
    print("Plotting histograms...")
    fig, axs = plt.subplots(1, 2, figsize=(25, 5))

    plot_multiple_hist(
        axs[0],
        arrays=[grad_og_edge, grad_og_mid],
        labels=["Gradient at Edges", "Gradients at middle of tiles"],
        colors=["blue", "black"],
        title="Original Method: Edge vs Middle Gradients"
    )

    plot_multiple_hist(
        axs[1],
        arrays=[grad_sw_mid, grad_sw_edge],
        labels=["Gradients at middle of tiles", "Gradient at Edges"],
        colors=["black", "red"],
        title="Sliding Window Method: Edge vs Middle Gradients"
    )

    plt.tight_layout()
    hist_path = os.path.join(args.save_dir, "histograms_gradients.png")
    plt.savefig(hist_path)
    plt.close(fig)
    print(f"Saved histogram plot: {hist_path}")

    # === Plot bar comparisons ===
    fig, ax = plt.subplots(4, 1, figsize=(25, 12))

    plot_multiple_bar(
        ax[0],
        arrays=[grad_utils_sw.histogram_edges, grad_utils_sw.histogram_middle],
        labels=["SW: Edge of Tiles", "SW: Middle of Tiles"],
        colors=["red", "black"],
        title="Sliding Window Gradients: Edge vs Middle",
        smooth_window=100,
        bin_edges=bin_edges[:-1]
    )

    plot_multiple_bar(
        ax[1],
        arrays=[grad_utils_og.histogram_edges, grad_utils_og.histogram_middle],
        labels=["OG: Edge of Tiles", "OG: Middle of Tiles"],
        colors=["blue", "black"],
        title="Original Image Gradients: Edge vs Middle",
        smooth_window=100,
        bin_edges=bin_edges[:-1]
    )

    plot_multiple_bar(
        ax[2],
        arrays=[
            grad_utils_og.histogram_middle - grad_utils_og.histogram_edges,
            grad_utils_sw.histogram_middle - grad_utils_sw.histogram_edges,
        ],
        labels=["OG: Middle - Edge", "SW: Middle - Edge"],
        colors=["blue", "red"],
        title="Histogram Differences (Middle minus Edge)",
        smooth_window=100,
        bin_edges=bin_edges[:-1],
    )

    plot_multiple_bar(
        ax[3],
        arrays=[
            grad_utils_sw.histogram_edges - grad_utils_og.histogram_edges,
            grad_utils_sw.histogram_middle - grad_utils_og.histogram_middle,
        ],
        labels=["Edge: SW - OG", "Middle: SW - OG"],
        colors=["orange", "black"],
        title="Histogram Differences Between SW and OG",
        smooth_window=100,
        bin_edges=bin_edges[:-1],
    )

    plt.tight_layout()
    bar_path = os.path.join(args.save_dir, "bar_comparisons.png")
    plt.savefig(bar_path)
    plt.close(fig)
    print(f"Saved bar comparison plot: {bar_path}")

    # === Plot KL Divergence Heatmaps ===
    print("Plotting KL divergence heatmaps...")
    plot_kl_heatmaps_for_range(
        grad_utils_list=[grad_utils_og, grad_utils_sw],
        bin_edges=bin_edges,
        start=args.kl_start,
        end=args.kl_end,
        channels=args.channel,
        labels=["OG", "SW"],
        save_dir=args.save_dir
    )

    print(f"\nAll analysis complete. Results saved in {args.save_dir}")


if __name__ == "__main__":
    main()
