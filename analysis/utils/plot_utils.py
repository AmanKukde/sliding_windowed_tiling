# plotting_utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
from matplotlib.gridspec import GridSpec
from microsplit_reproducibility.utils.paper_metrics import avg_range_inv_psnr
from microsplit_reproducibility.utils.paper_metrics import RangeInvariantPsnr
from utils.gradient_utils import GradientUtils
import pandas as pd
# --------------------------
# Gradient Histogram & KL Functions
# --------------------------
def normalize_histogram(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=float)
    return arr / (arr.sum() + eps)

def kl_divergence(p, q, eps=1e-12):
    p = normalize_histogram(p, eps)
    q = normalize_histogram(q, eps)
    return np.sum(p * np.log((p + eps) / (q + eps)))

def compute_kl_matrix(histograms):
    n = len(histograms)
    kl_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                kl_mat[i, j] = kl_divergence(histograms[i], histograms[j])
    return kl_mat

def plot_multiple_hist(ax, arrays, labels, colors, title, legend=False):
    if not (len(arrays) == len(labels) == len(colors)):
        raise ValueError("arrays, labels, and colors must have the same length")

    non_empty_arrays = [a for a in arrays if a.size > 0]
    if not non_empty_arrays:
        raise ValueError("All input arrays are empty, cannot plot histogram")

    all_min = min(np.min(a) for a in non_empty_arrays)
    all_max = max(np.max(a) for a in non_empty_arrays)
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

def plot_multiple_bar(ax, arrays, bin_edges, labels, colors, title, smooth_window=3, legend=True):
    n_bins = len(arrays[0])
    if len(bin_edges) != n_bins:
        raise ValueError("Length of bin_edges must match length of arrays")

    bar_width = np.min(np.diff(bin_edges)) * 0.4

    for i, (arr, label, color) in enumerate(zip(arrays, labels, colors)):
        ax.bar(bin_edges + i * bar_width, arr, width=bar_width, color=color, alpha=0.7, label=label)

        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            arr_smooth = np.convolve(arr, kernel, mode='same')
        else:
            arr_smooth = arr

        ax.plot(bin_edges, arr_smooth, color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Bin edges")
    ax.set_ylabel("Value")
    ax.grid(True, axis='y')
    if legend:
        ax.legend()

def plot_kl_heatmaps_for_range(grad_utils_list, bin_edges, start=29, end=34, channels=1, labels=None, cmap="coolwarm"):
    n_utils = len(grad_utils_list)
    if labels is None:
        labels = [f"Model{i}" for i in range(n_utils)]

    middle_hists = []
    for gu in grad_utils_list:
        grad_mid = gu.get_gradients_at("middle", channels=channels)
        middle_hists.append(GradientUtils.compute_histograms(grad_mid, bin_edges))

    n_plots = end - start + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 7.5), constrained_layout=False)
    if n_plots == 1:
        axes = [axes]

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

    for ax, index, kl_mat in zip(axes, range(start, end + 1), kl_mats):
        hist_labels = []
        for label in labels:
            hist_labels.extend([f"{label}-Edge", f"{label}-Mid"])
        sns.heatmap(kl_mat, annot=True, fmt=".3f", xticklabels=hist_labels,
                    yticklabels=hist_labels, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False, ax=ax)
        ax.set_title(f"Index {index}")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.8,
        label="KL Divergence"
    )
    fig.suptitle("KL Divergence Between Gradient Distributions", fontsize=16)
    # plt.show()

# --------------------------
# PSNR Functions
# --------------------------
def plot_per_image_psnr(df):
    for ch in df['Channel'].unique():
        sub_df = df[df['Channel'] == ch]
        plt.figure(figsize=(8, 4))
        plt.plot(sub_df['Image'], sub_df['PSNR_OG'], label='PSNR_OG', marker='o')
        plt.plot(sub_df['Image'], sub_df['PSNR_WIN'], label='PSNR_WIN', marker='x')
        plt.title(f'Per-Image PSNR for {ch}')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

def plot_avg_psnr_with_ci(avg_df, n_samples=6):
    t_crit = t.ppf(0.975, df=n_samples-1)
    ci_og = avg_df['SE_OG'] * t_crit
    ci_win = avg_df['SE_WIN'] * t_crit

    x = np.arange(len(avg_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, avg_df['Avg_PSNR_OG'], width, yerr=ci_og, capsize=5, label='OG')
    ax.bar(x + width/2, avg_df['Avg_PSNR_WIN'], width, yerr=ci_win, capsize=5, label='WIN')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['Channel'])
    ax.set_ylabel('Avg PSNR with 95% CI')
    ax.set_title('Mean PSNR with Confidence Intervals')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    # plt.show()

def plot_psnr_difference(avg_df,save_dir):
    delta = avg_df['Avg_PSNR_WIN'] - avg_df['Avg_PSNR_OG']
    x = np.arange(len(avg_df))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, delta, color='orange')
    plt.xticks(x, avg_df['Channel'])
    plt.ylabel('Δ PSNR (WIN - OG)')
    plt.title('PSNR Improvement of WIN over OG')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for bar, d in zip(bars, delta):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{d:.2f}', ha='center', va='bottom')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'PSNR_Improvement.png', dpi=300)
    # plt.show()

def plot_avg_psnr_zoomed(avg_df,save_dir):
    x = np.arange(len(avg_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, avg_df['Avg_PSNR_OG'], width, label='OG')
    ax.bar(x + width/2, avg_df['Avg_PSNR_WIN'], width, label='WIN')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['Channel'])
    ax.set_ylabel('Average PSNR')
    ax.set_title('Zoomed-In Average PSNR Comparison')
    ax.legend()
    y_min = min(avg_df[['Avg_PSNR_OG', 'Avg_PSNR_WIN']].min()) - 0.2
    y_max = max(avg_df[['Avg_PSNR_OG', 'Avg_PSNR_WIN']].max()) + 0.2
    ax.set_ylim([y_min, y_max])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'Zoomed_Avg_PSNR_Comparison.png', dpi=300)
    # plt.show()

def plot_all_psnr(avg_df, n_samples=6, save_dir=None):
    t_crit = t.ppf(0.975, df=n_samples - 1)

    # Safely get SE columns or fill with zeros if missing
    ci_og = avg_df.get('SE_OG', np.zeros(len(avg_df))) * t_crit
    ci_win = avg_df.get('SE_WIN', np.zeros(len(avg_df))) * t_crit

    x = np.arange(len(avg_df))
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    # Paired line plot
    for i in x:
        axs[0].plot([0, 1],
                    [avg_df['Avg_PSNR_OG'][i], avg_df['Avg_PSNR_WIN'][i]],
                    marker='o', linewidth=2)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(['OG', 'WIN'])
    axs[0].set_ylabel('Avg PSNR')
    axs[0].set_title('Paired Line Plot (OG vs WIN)')
    axs[0].grid(True, axis='y')

    # Error bar plot
    axs[1].errorbar(x - 0.05, avg_df['Avg_PSNR_OG'], yerr=ci_og, fmt='o', capsize=5, label='OG')
    axs[1].errorbar(x + 0.05, avg_df['Avg_PSNR_WIN'], yerr=ci_win, fmt='o', capsize=5, label='WIN')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(avg_df.get('Channel', [str(i) for i in x]))
    axs[1].set_ylabel('Avg PSNR')
    axs[1].set_title('Dot Plot with 95% Confidence Interval')
    axs[1].legend()
    axs[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'All_PSNR_Plots.png', dpi=300)

def to_scalar_tuple(x):
    if isinstance(x, (tuple, list)):
        return float(x[0]), float(x[1])
    elif hasattr(x, 'item'):
        return float(x.item()), 0.0
    return float(x), 0.0

def compute_psnr_and_plot(test_data, img_og, img_win, save_dir):


    # --- Per-image PSNR ---
    records = []
    for i in range(min(6, len(test_data))):
        tar = test_data[i:i+1]
        inp_win = img_win[i:i+1]
        inp_og = img_og[i:i+1]

        for ch in range(2):
            psnr_og, _ = to_scalar_tuple(RangeInvariantPsnr(tar[..., ch], inp_og[..., ch]))
            psnr_win, _ = to_scalar_tuple(RangeInvariantPsnr(tar[..., ch], inp_win[..., ch]))

            records.append({
                'Image': i,
                'Channel': f'Ch{ch+1}',
                'PSNR_OG': psnr_og,
                'PSNR_WIN': psnr_win
            })

    psnr_df = pd.DataFrame(records)

    # --- Average PSNR + STD ---
    avg_records = []
    for ch in [0, 1]:
        avg_og, std_og = to_scalar_tuple(avg_range_inv_psnr(test_data[..., ch], img_og[..., ch]))
        avg_win, std_win = to_scalar_tuple(avg_range_inv_psnr(test_data[..., ch], img_win[..., ch]))

        avg_records.append({
            'Channel': f'Ch{ch+1}',
            'Avg_PSNR_OG': avg_og,
            'Std_OG': std_og,
            'Avg_PSNR_WIN': avg_win,
            'Std_WIN': std_win
        })

    avg_df = pd.DataFrame(avg_records)

    # --- Plot: Per-image PSNR ---
    for ch in ['Ch1', 'Ch2']:
        sub_df = psnr_df[psnr_df['Channel'] == ch]
        plt.figure(figsize=(8, 4))
        plt.plot(sub_df['Image'], sub_df['PSNR_OG'], label='PSNR_OG', marker='o')
        plt.plot(sub_df['Image'], sub_df['PSNR_WIN'], label='PSNR_WIN', marker='x')
        plt.title(f'Per-Image PSNR for {ch}')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'Per_Image_PSNR_{ch}.png', dpi=300)
        # plt.show()

    # --- Plot: Average PSNR with Std ---
    x = np.arange(len(avg_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, avg_df['Avg_PSNR_OG'], width, yerr=avg_df['Std_OG'], capsize=5, label='OG')
    ax.bar(x + width/2, avg_df['Avg_PSNR_WIN'], width, yerr=avg_df['Std_WIN'], capsize=5, label='WIN')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['Channel'])
    ax.set_ylabel('Average PSNR')
    ax.set_title('Average PSNR with StdDev by Channel')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'Avg_PSNR_with_Std.png', dpi=300)
    # plt.show()

    return psnr_df, avg_df


# --- full_frame_evaluation function (provided in your initial prompt) ---
def full_frame_evaluation(
    predictions_list,
    tar_list,
    inp_list,
    metrics_list,
    frame_idx,
    titles,
    save_path=None
):
    """
    Plots the input, target, and prediction images along with associated metrics
    for a given frame, comparing different evaluation methods.
    Custom layout:
    Row 1: [Empty] [Input Image] [Empty] | [Empty] [Input Image] [Empty]
    Row 2: Target Ch0 | Target Ch1 | Target Ch0 | Target Ch1
    Row 3: Pred Ch0 | Pred Ch1 | Pred Ch0 | Pred Ch1
    Row 4: Metrics Ch0 | Metrics Ch1 | Metrics Ch0 | Metrics Ch1

    Args:
        predictions_list (list): A list of prediction arrays. Each array
                                 should have a shape compatible with (..., H, W, 2).
        tar_list (list): A list of target arrays. Each array should have a
                         shape compatible with (..., H, W, 2).
        inp_list (list): A list of input arrays. Each array should have a
                         shape compatible with (..., H, W) (grayscale).
        metrics_list (list): A list of dictionaries, where each dictionary
                             contains metrics for a specific evaluation method.
                             Metrics are expected to be per-channel, e.g.,
                             {'metric_name': [(value_ch0, 0.0), (value_ch1, 0.0)]}.
        frame_idx (int): The index of the current frame being evaluated.
        titles (list): A list of strings, where each string is the title
                       for a set of results (e.g., ["windowed", "MMSE 64"]).
        save_path (str, optional): If provided, the plot will be saved to this path.
                                   Defaults to None (plot is displayed).
    """
    
    fig = plt.figure(figsize=(24, 20)) 

    gs_main = GridSpec(4, 6, figure=fig, wspace=0.05, hspace=0.15, 
                    height_ratios=[1.2, 1, 1, 0.4]) 

    metrics_data = [] 

    for metrics_dict in metrics_list:
        method_metrics = {'ch0': {}, 'ch1': {}}
        for metric_name, values in metrics_dict.items():
            method_metrics['ch0'][metric_name] = values[0][0]
            method_metrics['ch1'][metric_name] = values[1][0]
        metrics_data.append(method_metrics)

    print(" --- Row 1 (gs_main[0, :]): Input Images (centered) ---")
    for i in range(2): 
        col_offset = i * 3 
        print(f"Plotting input for method {titles[i]} at column offset {col_offset}")
        current_inp = (inp_list[i][...,0] + inp_list[i][...,1])/2.  # Accessing the correct frame_idx for input

        gs_input_section = gs_main[0, col_offset:col_offset+3].subgridspec(1, 1, wspace=0.05, hspace=0.05)
        
        ax_input = fig.add_subplot(gs_input_section[0, 0])
        ax_input.imshow(current_inp)
        ax_input.set_title(f"{titles[i]}\nInput (Frame {frame_idx})")
        ax_input.axis('off')

    current_target = tar_list[0] # Accessing the correct frame_idx for target

    current_prediction_win = predictions_list[0]
    current_prediction_og = predictions_list[1]

    print("--- Row 2 (gs_main[1, :]): Target Channels ---")
    gs_targets = gs_main[1, :].subgridspec(1, 4, wspace=0.05, hspace=0.05)
    ax_tar_win_ch0 = fig.add_subplot(gs_targets[0, 0])
    ax_tar_win_ch0.imshow(current_target[..., 0])
    ax_tar_win_ch0.set_title(f"{titles[0]}\nTarget Ch0")
    ax_tar_win_ch0.axis('off')

    ax_tar_og_ch0 = fig.add_subplot(gs_targets[0, 1])
    ax_tar_og_ch0.imshow(current_target[..., 0])
    ax_tar_og_ch0.set_title(f"{titles[1]}\nTarget Ch0")
    ax_tar_og_ch0.axis('off')

    ax_tar_win_ch1 = fig.add_subplot(gs_targets[0, 2])
    ax_tar_win_ch1.imshow(current_target[..., 1])
    ax_tar_win_ch1.set_title(f"{titles[0]}\nTarget Ch1")
    ax_tar_win_ch1.axis('off')

    ax_tar_og_ch1 = fig.add_subplot(gs_targets[0, 3])
    ax_tar_og_ch1.imshow(current_target[..., 1])
    ax_tar_og_ch1.set_title(f"{titles[1]}\nTarget Ch1")
    ax_tar_og_ch1.axis('off')

    print("--- Row 3 (gs_main[2, :]): Prediction Channels ---")
    gs_preds = gs_main[2, :].subgridspec(1, 4, wspace=0.05, hspace=0.05)
    ax_pred_win_ch0 = fig.add_subplot(gs_preds[0, 0])
    ax_pred_win_ch0.imshow(current_prediction_win[..., 0])
    ax_pred_win_ch0.set_title(f"{titles[0]}\nPrediction Ch0")
    ax_pred_win_ch0.axis('off')

    ax_pred_og_ch0 = fig.add_subplot(gs_preds[0, 1])
    ax_pred_og_ch0.imshow(current_prediction_og[..., 0])
    ax_pred_og_ch0.set_title(f"{titles[1]}\nPrediction Ch0")
    ax_pred_og_ch0.axis('off')

    ax_pred_win_ch1 = fig.add_subplot(gs_preds[0, 2])
    ax_pred_win_ch1.imshow(current_prediction_win[..., 1])
    ax_pred_win_ch1.set_title(f"{titles[0]}\nPrediction Ch1")
    ax_pred_win_ch1.axis('off')

    ax_pred_og_ch1 = fig.add_subplot(gs_preds[0, 3])
    ax_pred_og_ch1.imshow(current_prediction_og[..., 1])
    ax_pred_og_ch1.set_title(f"{titles[1]}\nPrediction Ch1")
    ax_pred_og_ch1.axis('off')
    
    # --- Row 4 (gs_main[3, :]): Metrics ---
    gs_metrics = gs_main[3, :].subgridspec(1, 4, wspace=0.05, hspace=0.05)
    
    metric_axes = [
        fig.add_subplot(gs_metrics[0, 0]), 
        fig.add_subplot(gs_metrics[0, 1]), 
        fig.add_subplot(gs_metrics[0, 2]), 
        fig.add_subplot(gs_metrics[0, 3])  
    ]

    for ax in metric_axes:
        ax.axis('off') 

    def plot_metrics_with_bolding(ax, method_idx, channel_key, title):
        ax.set_title(title)
        
        metric_names = sorted(metrics_data[method_idx][channel_key].keys())
        
        y_position = 0.5 
        line_spacing = 1.0 / (len(metric_names) + 1) 
        
        bbox_height_per_line = 0.22 
        total_bbox_height = len(metric_names) * bbox_height_per_line

        rect = plt.Rectangle((0, 0.5 - total_bbox_height/2), 1, total_bbox_height, 
                            transform=ax.transAxes,
                            facecolor='lightgray', edgecolor='k', linewidth=1, alpha=0.7)
        ax.add_patch(rect)

        for i, metric_name in enumerate(metric_names):
            current_value = metrics_data[method_idx][channel_key][metric_name]
            other_method_idx = 1 if method_idx == 0 else 0
            other_value = metrics_data[other_method_idx][channel_key][metric_name]

            text_weight = 'normal'
            # Only bold if the current value is strictly greater than the other value
            if round(current_value, 2) > round(other_value, 2):
                text_weight = 'bold'
            
            ax.text(0.5, y_position + ( (len(metric_names) - 1) / 2 - i) * line_spacing,
                    f"{metric_name}: {current_value:.2f}",
                    verticalalignment='center', horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    weight=text_weight) 

    # Plotting for Windowed (Method 0)
    plot_metrics_with_bolding(metric_axes[0], 0, 'ch0', f"")
    plot_metrics_with_bolding(metric_axes[1], 1, 'ch0', f"") 
    plot_metrics_with_bolding(metric_axes[2], 0, 'ch1', f"") 
    plot_metrics_with_bolding(metric_axes[3], 1, 'ch1', f"") 


    plt.tight_layout() 

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
