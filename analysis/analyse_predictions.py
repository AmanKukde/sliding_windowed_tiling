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
        default="gradient_analysis_results_16Oct25",
        help="Where to save analysis results.",
    )
    parser.add_argument(
        "--python_bin",
        default="/scratch/aman.kukde/conda/envs/msr/bin/python3.10",
        help="Python interpreter to use (e.g. /path/to/conda/env/bin/python).",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="How many analyses to run in parallel.")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be run without executing.")
    return parser.parse_args()


def find_prediction_pairs(results_base: Path):
    """
    Find all SW–OG prediction file pairs.
    Assumes folder structure: <dataset>/<modality>/<lc>/ with *_og.* and *_stitched.* files
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

                og_files = list(lc_dir.glob("*_og.*"))
                sw_files = list(lc_dir.glob("*_stitched.*"))

                if og_files and sw_files:
                    pairs.append({
                        "dataset": dataset_dir.name,
                        "modality": modality_dir.name,
                        "lc": lc_dir.name,
                        "og_path": str(og_files[0]),
                        "sw_path": str(sw_files[0]),
                    })
    return pairs


def run_analysis(pair, args, analyze_script: Path):
    dataset, modality, lc = pair["dataset"], pair["modality"], pair["lc"]
    sw_path, og_path = pair["sw_path"], pair["og_path"]

    save_dir = Path(args.save_base) / dataset / modality / lc
    save_dir.mkdir(parents=True, exist_ok=True)

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

    log_entry = [f"=== {dataset}/{modality}/{lc} ==="]

    if args.dry_run:
        msg = "[DRY RUN] " + " ".join(cmd)
        print(msg)
        log_entry.append(msg)
        return "\n".join(log_entry)

    print(f"▶ Running gradient analysis for {dataset} / {modality} / {lc}")

    if not analyze_script.exists():
        msg = f"[MISSING SCRIPT] {analyze_script}"
        print(f"❌ {msg}")
        log_entry.append(msg)
        return "\n".join(log_entry)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Completed {dataset} / {modality} / {lc}")
        log_entry.append("[DONE]")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed {dataset}/{modality}/{lc}")
        log_entry.append("[FAILED]")
        log_entry.append(f"Command: {' '.join(cmd)}")
        log_entry.append(f"Exit Code: {e.returncode}")
        log_entry.append(f"Stdout:\n{e.stdout.strip()}")
        log_entry.append(f"Stderr:\n{e.stderr.strip()}")
    except Exception as e:
        log_entry.append(f"[ERROR] Unexpected exception: {repr(e)}")

    return "\n".join(log_entry)


def main():
    args = parse_args()
    results_base = Path(args.results_base)
    save_base = Path(args.save_base)
    save_base.mkdir(parents=True, exist_ok=True)

    analyze_script = Path(args.project_dir) / "analysis" / "analyze_gradients.py"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_base / f"run_log_{timestamp}.txt"

    print(f"🔍 Scanning for prediction pairs in {results_base}")
    pairs = find_prediction_pairs(results_base)
    print(f"Found {len(pairs)} valid SW–OG pairs.\n")

    if not pairs:
        print("No matching prediction pairs found. Exiting.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_analysis, p, args, analyze_script) for p in pairs]
        for f in as_completed(futures):
            results.append(f.result())

    print("\nSummary:")
    for r in results:
        status_line = r.splitlines()[-1]
        print(" ", status_line)

    with open(log_file, "w") as f:
        f.write("\n\n".join(results))

    print(f"\n📝 Detailed log saved to {log_file}")


if __name__ == "__main__":
    main()
