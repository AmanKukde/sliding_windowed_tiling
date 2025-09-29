
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from gradient_utils import GradientUtils


def plot_multiple_hist(ax, arrays, labels, colors, title,legend = False):
    """
    Plot histograms + Gaussian fits for multiple arrays on the same axes.

    Args:
        ax : matplotlib axes object
        arrays : list of 1D numpy arrays
        labels : list of strings for each array
        colors : list of colors for each array
        title : string for the plot title
    """
    if not (len(arrays) == len(labels) == len(colors)):
        raise ValueError("arrays, labels, and colors must have the same length")

    # Determine global range
    all_min = min(np.min(a) for a in arrays)
    all_max = max(np.max(a) for a in arrays)
    if all_min == all_max:
        all_min -= 1e-3
        all_max += 1e-3

    x = np.linspace(all_min, all_max, 1000)

    for a, label, color in zip(arrays, labels, colors):
        mu, std = norm.fit(a)
        ax.hist(a, bins=100, density=True, alpha=0.5, label=label, color=color)
        ax.plot(x, norm.pdf(x, mu, std), linestyle='-', color=color,
                label=f'{label}\nFit μ={mu:.2f}, σ={std:.2f}')

    ax.set_title(title)
    ax.set_xlabel("Gradient Value")
    ax.set_ylabel("Density")
    ax.grid(True)
    if legend:
        ax.legend()


def plot_multiple_bar(ax, arrays, bin_edges, labels, colors, title, smooth_window=3, legend = True):
    """
    Plot multiple arrays as grouped bars on the same axes, optionally smoothing
    to show the general shape of the histograms.

    Args:
        ax : matplotlib axes object
        arrays : list of 1D numpy arrays of same length
        bin_edges : 1D numpy array of bin edges (length should match arrays)
        labels : list of strings for each array
        colors : list of colors for each array
        title : string for the plot title
        smooth_window : int, window size for moving average smoothing
    """
    n_bins = len(arrays[0])
    if len(bin_edges) != n_bins:
        raise ValueError("Length of bin_edges must match length of arrays")

    bar_width = np.min(np.diff(bin_edges)) * 0.4  # scale width based on spacing

    for i, (arr, label, color) in enumerate(zip(arrays, labels, colors)):
        # Offset each array slightly for grouped bars
        ax.bar(bin_edges + i * bar_width, arr, width=bar_width, color=color, alpha=0.7, label=label)

        # Compute a simple moving average for visualization
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            arr_smooth = np.convolve(arr, kernel, mode='same')
        else:
            arr_smooth = arr

        # Plot the smoothed line over the bars
        ax.plot(bin_edges, arr_smooth, color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Bin edges")
    ax.set_ylabel("Value")
    ax.grid(True, axis='y')
    if legend:
        ax.legend()


def normalize_histogram(arr, eps=1e-12):
    """
    Normalize a histogram to sum to 1 (probability distribution).
    
    Args:
        arr : 1D numpy array
        eps : small constant to avoid division by zero
    
    Returns:
        Normalized array
    """
    arr = np.asarray(arr, dtype=float)
    return arr / (arr.sum() + eps)

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL divergence D_KL(P || Q)
    
    Args:
        p, q : 1D numpy arrays representing probability distributions
        eps : small constant to avoid log(0)
    
    Returns:
        KL divergence scalar
    """
    p = normalize_histogram(p, eps)
    q = normalize_histogram(q, eps)
    return np.sum(p * np.log((p + eps) / (q + eps)))

def compute_kl_matrix(histograms):
    """
    Compute KL divergence matrix between multiple histograms.
    
    Args:
        histograms : list of 1D numpy arrays
    
    Returns:
        n x n numpy array of KL divergences
    """
    n = len(histograms)
    kl_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                kl_mat[i, j] = kl_divergence(histograms[i], histograms[j])
    return kl_mat
def plot_kl_heatmaps_for_range(grad_utils_list, bin_edges, start=29, end=34, channels=1, labels=None, cmap="coolwarm"):
    """
    Compute and plot KL divergence heatmaps for gradients at given index range
    compared to middle gradients, for multiple GradientUtils objects.

    Args:
        grad_utils_list : list of GradientUtils instances (e.g. [grad_utils_og, grad_utils_sw])
        bin_edges       : bin edges for histograms
        start           : start index (inclusive)
        end             : end index (inclusive)
        channels        : channel(s) to extract
        labels          : optional list of names for grad_utils (len must match grad_utils_list)
        cmap            : colormap for heatmaps
    """
    n_utils = len(grad_utils_list)
    if labels is None:
        labels = [f"Model{i}" for i in range(n_utils)]

    # Precompute "middle" histograms for each grad_utils
    middle_hists = []
    for gu in grad_utils_list:
        grad_mid = gu.get_gradients_at("middle", channels=channels)
        middle_hists.append(GradientUtils.compute_histograms(grad_mid, bin_edges))

    n_plots = end - start + 1

    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots,7.5), constrained_layout=False)

    if n_plots == 1:
        axes = [axes]  # make iterable

    # Collect all KL matrices to determine global vmin/vmax for shared colorbar
    kl_mats = []
    for index in range(start, end + 1):
        histograms = []
        for gu, mid_hist in zip(grad_utils_list, middle_hists):
            grad_at_idx = gu.get_gradients_at(index, channels=channels)
            hist_at_idx = GradientUtils.compute_histograms(grad_at_idx, bin_edges)
            histograms.extend([hist_at_idx, mid_hist])
        kl_mats.append(compute_kl_matrix(histograms))

    vmin = min(np.min(mat) for mat in kl_mats)
    vmax = max(np.max(mat) for mat in kl_mats)

    # Plot heatmaps
    for ax, index, kl_mat in zip(axes, range(start, end + 1), kl_mats):
        hist_labels = []
        for label in labels:
            hist_labels.extend([f"{label}-Edge", f"{label}-Mid"])

        sns.heatmap(
            kl_mat,
            annot=True,
            fmt=".3f",
            xticklabels=hist_labels,
            yticklabels=hist_labels,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar=False,   # disable individual colorbars
            ax=ax
        )
        ax.set_title(f"Index {index}")

    # Shared colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.8,
        label="KL Divergence"
    )

    fig.suptitle("KL Divergence Between Gradient Distributions", fontsize=16)
    plt.show()
