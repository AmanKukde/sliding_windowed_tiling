#!/usr/bin/env python3
"""
Batch Gradient Analysis Runner
------------------------------
Runs analyze_gradients.py for all available dataset/modality/LC combinations
where both SW (sliding window) and OG (original) predictions exist.
Automatically detects SW and OG files in the current folder structure.
"""

import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradient analysis on multiple results.")
    parser.add_argument(
        "--results_base",
        default="/group/jug/aman/usplit_13Oct25/",
        help="Base directory containing prediction results for all datasets.",
    )
    parser.add_argument(
        "--project_dir",
        default="/home/aman.kukde/sliding_windowed_tiling/",
        help="Root of your project (should contain analyze_gradients.py).",
    )
    parser.add_argument(
        "--save_base",
        default="gradient_analysis_results_local",
        help="Where to save analysis results.",
    )
    parser.add_argument(
        "--python_bin",
        default="/localscratch/conda/envs/msr/bin/python3.10",
        help="Python interpreter to use (e.g. /path/to/conda/env/bin/python).",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="How many analyses to run in parallel.")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be run without executing.")
    return parser.parse_args()


def find_prediction_pairs(results_base: Path):
    """
    Find all SW–OG prediction file pairs.
    Assumes folder structure: <dataset>/<modality>/<lc>/ with *_og.* and *_windowed.* files
    """
    pairs = []

    for dataset_dir in results_base.iterdir():
        if not dataset_dir.is_dir():
            continue
        for modality_dir in dataset_dir.iterdir():
            if not modality_dir.is_dir():
                continue
            for lc_dir in modality_dir.iterdir():
                if not lc_dir.is_dir():
                    continue

                # Look for *_og.* and *_windowed.* files
                og_files = list(lc_dir.glob("*_og.pkl*"))
                sw_files = list(lc_dir.glob("*_stitched.pkl*"))

                if og_files and sw_files:
                    # Take the first file of each type (assuming one pair per LC folder)
                    pairs.append({
                        "dataset": dataset_dir.name,
                        "modality": modality_dir.name,
                        "lc": lc_dir.name,
                        "og_path": str(og_files[0]),
                        "sw_path": str(sw_files[0]),
                    })

    return pairs


def run_analysis(pair, args):
    dataset = pair["dataset"]
    modality = pair["modality"]
    lc = pair["lc"]
    sw_path = pair["sw_path"]
    og_path = pair["og_path"]

    save_dir = Path(args.save_base) / dataset / modality / lc
    save_dir.mkdir(parents=True, exist_ok=True)

    analyze_script = Path(args.project_dir) / "analysis" / "analyze.py"
    cmd = [
        args.python_bin,
        str(analyze_script),
        "--sw_path", sw_path,
        "--og_path", og_path,
        "--save_dir", str(save_dir),
        "--tile_size", "32",
        "--bins", "2000",
        "--channel", "1",
        "--kl_start", "29",
        "--kl_end", "33",
    ]

    if args.dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return f"[SKIPPED] {dataset}/{modality}/{lc}"

    print(f"▶ Running analysis for {dataset} / {modality} / {lc}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Completed {dataset} / {modality} / {lc}")
        return f"[DONE] {dataset}/{modality}/{lc}"
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed {dataset}/{modality}/{lc}: {e}")
        return f"[FAILED] {dataset}/{modality}/{lc}"


def main():
    args = parse_args()
    results_base = Path(args.results_base)
    save_base = Path(args.save_base)
    save_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_base / f"run_log_{timestamp}.txt"

    print(f"🔍 Scanning for prediction pairs in {results_base}")
    pairs = find_prediction_pairs(results_base)
    print(f"Found {len(pairs)} valid SW–OG pairs.")

    if not pairs:
        print("No matching prediction pairs found. Exiting.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_analysis, p, args) for p in pairs]
        for f in as_completed(futures):
            results.append(f.result())

    print("\nSummary:")
    for r in results:
        print(" ", r)

    with open(log_file, "w") as f:
        f.write("\n".join(results))
    print(f"\n📝 Log saved to {log_file}")


if __name__ == "__main__":
    main()
