# analysis_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from usplit.core.psnr import RangeInvariantPsnr as psnr
from utils.gradient_utils import GradientUtils2D as GradientUtils
from utils.plot_utils import (
    plot_multiple_hist,
    plot_multiple_bar,
    plot_kl_heatmaps_for_range,
    normalize_histogram,
    compute_kl_matrix,
)

# -------------------------------
# Basic metric utilities
# -------------------------------

def compute_psnr(pred1, pred2):
    """Compute PSNR between two numpy arrays."""
    return psnr(pred1, pred2, data_range=pred1.max() - pred1.min())

def compute_peakiness(hist, bin_edges):
    """
    Calculates the 'peakiness' of a histogram, defined as the sum of the top 10% bin masses after normalization.

    Parameters:
        hist (array-like): The histogram bin counts.
        bin_edges (array-like): The edges of the histogram bins (unused in calculation).

    Returns:
        float: The sum of the top 10% normalized bin masses, representing the histogram's peakiness.
    """
    """Measure 'peakiness' of a histogram = ratio of top 10% bin mass to total."""
    hist = normalize_histogram(hist)
    sorted_vals = np.sort(hist)[::-1]
    top_frac = int(0.1 * len(sorted_vals))
    return np.sum(sorted_vals[:top_frac])

def summarize_gradients(grad_utils_og, grad_utils_sw, bin_edges, channel, save_dir):
    """Generate and save key gradient visualizations + metrics."""
    os.makedirs(save_dir, exist_ok=True)

    # Extract gradients at edge and middle positions
    grad_edge_og = grad_utils_og.get_gradients_at("edge", channels=channel)
    grad_mid_og = grad_utils_og.get_gradients_at("middle", channels=channel)
    grad_edge_sw = grad_utils_sw.get_gradients_at("edge", channels=channel)
    grad_mid_sw = grad_utils_sw.get_gradients_at("middle", channels=channel)

    # Compute histograms
    h_edge_og = GradientUtils.compute_histograms(grad_edge_og, bin_edges)
    h_mid_og = GradientUtils.compute_histograms(grad_mid_og, bin_edges)
    h_edge_sw = GradientUtils.compute_histograms(grad_edge_sw, bin_edges)
    h_mid_sw = GradientUtils.compute_histograms(grad_mid_sw, bin_edges)


    fig, axs = plt.subplots(1, 2, figsize=(25, 5))
    
    plot_multiple_hist(
        axs[0],
        arrays=[grad_edge_og, grad_mid_og],
        labels=["Gradient at Edges", "Gradients at middle of tiles"],
        colors=["blue", "black"],
        title="Gradients of Original vs In the Middle of Tiles",
        legend=True
    )
    
    plot_multiple_hist(
        axs[1],
        arrays=[grad_mid_sw, grad_edge_sw],
        labels=["Gradients at middle of tiles", "Gradient at Edges"],
        colors=["black", "red"],
        title="Gradients of SW vs In the Middle of Tiles",
        legend=True
    )
    
    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_histograms_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2ï¸âƒ£ Plot bar charts with differences (like your notebook)
    fig, ax = plt.subplots(4, 1, figsize=(17, 12))
    
    plot_multiple_bar(
        ax[0],
        arrays=[h_edge_sw, h_mid_sw],
        labels=["SW: Edge of Tiles", "SW: Middle of Tiles"],
        colors=["red", "black"],
        title="Sliding Window Gradients: Edge vs Middle",
        smooth_window=25,
        bin_edges=bin_edges[:-1]
    )
    
    plot_multiple_bar(
        ax[1],
        arrays=[h_edge_og, h_mid_og],
        labels=["OG: Edge of Tiles", "OG: Middle of Tiles"],
        colors=["blue", "black"],
        title="Original Image Gradients: Edge vs Middle",
        smooth_window=25,
        bin_edges=bin_edges[:-1]
    )
    
    plot_multiple_bar(
        ax[2],
        arrays=[h_mid_og - h_edge_og, h_mid_sw - h_edge_sw],
        labels=["OG: Middle - Edge", "SW: Middle - Edge"],
        colors=["blue", "red"],
        title="Histogram Differences (Middle minus Edge) per Image",
        smooth_window=25,
        bin_edges=bin_edges[:-1]
    )
    
    plot_multiple_bar(
        ax[3],
        arrays=[h_edge_sw - h_edge_og, h_mid_sw - h_mid_og],
        labels=["Edge: SW - OG", "Middle: SW - OG"],
        colors=["orange", "black"],
        title="Histogram Differences Between SW and OG at Tile Positions",
        smooth_window=25,
        bin_edges=bin_edges[:-1]
    )
    
    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_bar_charts.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3ï¸âƒ£ KL divergence heatmaps
    fig_kl = plot_kl_heatmaps_for_range(
        [grad_utils_og, grad_utils_sw],
        bin_edges,
        start=29,
        end=33,
        channels=channel,
        labels=["OG", "SW"],
    )
    if fig_kl is not None:
        fig_kl.savefig(Path(save_dir) / "kl_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close(fig_kl)

    # 4ï¸âƒ£ Compute peakiness scores
    peakiness_og = grad_utils_og.get_peakiness_scores(h_edge_og, h_mid_og)[-1]
    peakiness_sw = grad_utils_sw.get_peakiness_scores(h_edge_sw, h_mid_sw)[-1]
    peakiness_delta =  peakiness_og - peakiness_sw

    # 5ï¸âƒ£ KL divergence summary
    kl_edge_mid_og = compute_kl_matrix([h_edge_og, h_mid_og])
    kl_edge_mid_sw = compute_kl_matrix([h_edge_sw, h_mid_sw])

    # Write out summary
    summary_path = Path(save_dir) / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Gradient Analysis Summary ===\n\n")
        f.write("Peakiness Scores (Lower is better):\n")
        f.write(f"  Original Method: {peakiness_og:.6f}\n")
        f.write(f"  Sliding Window Method: {peakiness_sw:.6f}\n")
        f.write(f"  Î” (OG - SW): {peakiness_delta:.6f}\n")
        if peakiness_delta > 0:
            f.write("  ✅ Sliding Window Method performs better (lower peakiness)\n")
        else:
            f.write("  âŒ Original Method performs better (lower peakiness)\n")
        f.write(f"\nKL Divergence OG (edge vs mid): {kl_edge_mid_og[0,1]:.6f}\n")
        f.write(f"KL Divergence SW (edge vs mid): {kl_edge_mid_sw[0,1]:.6f}\n")

    print(f"✅ Gradient analysis summary saved to {summary_path}")


# -------------------------------
# PSNR + reconstruction helpers
# -------------------------------

def compute_psnr_and_plot(pred_sw, pred_og, target, save_dir):
    """
    Compare SW vs OG predictions against target and save PSNR plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    psnr_og = compute_psnr(pred_og, target)
    psnr_sw = compute_psnr(pred_sw, target)
    delta = psnr_sw - psnr_og

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["OG", "SW"], [psnr_og, psnr_sw], color=["gray", "orange"])
    ax.set_title(f"PSNR Comparison (Î”={delta:.2f})")
    ax.set_ylabel("PSNR (dB)")
    fig.savefig(Path(save_dir) / "psnr_comparison.png", dpi=300)
    plt.close(fig)

    with open(Path(save_dir) / "psnr_summary.txt", "w") as f:
        f.write(f"OG PSNR: {psnr_og:.4f}\nSW PSNR: {psnr_sw:.4f}\nÎ”: {delta:.4f}\n")

    print(f"✅ PSNR analysis saved to {save_dir}")