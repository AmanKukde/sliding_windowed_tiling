# analyze_gradients.py
import argparse
import pickle
import tifffile as tiff
from pathlib import Path
from utils.gradient_utils import GradientUtils
from utils.analysis_utils import summarize_gradients, compute_psnr_and_plot


def main():
    parser = argparse.ArgumentParser(description="Run gradient + PSNR analysis on one experiment.")
    parser.add_argument("--sw_path", required=True, help="Path to sliding window prediction (pkl or tiff)")
    parser.add_argument("--og_path", required=True, help="Path to original prediction (pkl or tiff)")
    parser.add_argument("--save_dir", required=True, help="Directory to store results")
    parser.add_argument("--tile_size", type=int, default=32)
    parser.add_argument("--bins", type=int, default=2000)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--kl_start", type=int, default=29)
    parser.add_argument("--kl_end", type=int, default=33)
    parser.add_argument("--target_path", default=None, help="Optional target GT for PSNR")

    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load data (support .tiff and .pkl)
    def load_pred(path):
        path = Path(path)
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            return tiff.imread(path)

    pred_sw = load_pred(args.sw_path)
    pred_og = load_pred(args.og_path)

    grad_utils_sw = GradientUtils(pred_sw, tile_size=args.tile_size)
    grad_utils_og = GradientUtils(pred_og, tile_size=args.tile_size)
    bin_edges = grad_utils_sw.make_bin_edges(n_bins=args.bins)

    # Run gradient comparison
    summarize_gradients(
        grad_utils_og,
        grad_utils_sw,
        bin_edges,
        channel=args.channel,
        save_dir=args.save_dir,
    )

    # Optional PSNR analysis if target available
    if args.target_path:
        target = load_pred(args.target_path)
        compute_psnr_and_plot(pred_sw, pred_og, target, args.save_dir)


if __name__ == "__main__":
    main()
