#!/usr/bin/env python3
"""
Interactive Batch Experiment Analysis Runner with HPC support
-------------------------------------------------------------
Allows user to select which SW–OG pairs to run. Can run locally
or submit jobs to HPC using SLURM when --hpc is specified.
"""

import os
import pdb
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch analysis on multiple results.")
    parser.add_argument("--results_base", default="/group/jug/aman/usplit_13Oct25/")
    parser.add_argument("--project_dir", default="/home/aman.kukde/sliding_windowed_tiling/")
    parser.add_argument("--save_base", default="/group/jug/aman/usplit_analysis_results_16Oct25/")
    parser.add_argument("--python_bin", default="/scratch/aman.kukde/conda/envs/msr/bin/python3.10")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--gradient_based_analysis", action="store_true")
    parser.add_argument("--qualitative_analysis", action="store_true")
    parser.add_argument("--all", action="store_false")
    parser.add_argument("--hpc", action="store_false", help="Submit analysis jobs to HPC using SLURM")
    parser.add_argument("--partition", default="gpuq", help="SLURM partition for HPC jobs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--mem", default="64GB")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--time", default="12:00:00")
    return parser.parse_args()

def find_prediction_pairs(results_base: Path):
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
                og_files = list(lc_dir.glob("*_og.pkl*"))
                sw_files = list(lc_dir.glob("*_stitched.pkl*"))
                if og_files and sw_files:
                    pairs.append({
                        "dataset": dataset_dir.name,
                        "modality": modality_dir.name,
                        "lc": lc_dir.name,
                        "og_path": str(og_files[0]),
                        "sw_path": str(sw_files[0]),
                    })
    return pairs

def select_pairs_interactively(pairs):
    print("\nAvailable SW–OG prediction pairs:")
    for i, p in enumerate(pairs, 1):
        print(f"{i}: {p['dataset']}/{p['modality']}/{p['lc']}")
    selection = input("\nEnter numbers of the pairs to run (comma-separated) or [A] for all:\n")
    if selection.strip().upper() == 'A':
        return pairs
    indices = [int(s.strip()) - 1 for s in selection.split(",") if s.strip().isdigit()]
    selected_pairs = [pairs[i] for i in indices if 0 <= i < len(pairs)]
    if not selected_pairs:
        print("No valid selections made. Exiting.")
        exit(0)
    return selected_pairs

def run_local(pair, args):
    dataset, modality, lc = pair["dataset"], pair["modality"], pair["lc"]
    sw_path, og_path = pair["sw_path"], pair["og_path"]
    save_dir = Path(args.save_base) / dataset / modality / lc
    save_dir.mkdir(parents=True, exist_ok=True)

    analyze_script = Path(args.project_dir) / "analysis" / "analyze_experiment.py"
    cmd = [
        args.python_bin, str(analyze_script),
        "--dataset", dataset,
        "--pred_sw", sw_path,
        "--pred_og", og_path,
        "--save_dir", str(save_dir),
        "--tile_size", "32",
        "--bins", "200",
        "--channel", "all",
        "--kl_start", "29",
        "--kl_end", "33",
    ]
    if args.gradient_based_analysis: cmd.append("--gradient_based_analysis")
    if args.qualitative_analysis: cmd.append("--qualitative_analysis")
    if args.all: cmd.append("--all")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_dir / f"analysis_log_{timestamp}.log"

    if args.dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return f"[SKIPPED] {dataset}/{modality}/{lc}"

    print(f"▶ Running analysis for {dataset}/{modality}/{lc}")
    try:
        with open(log_file, "w") as f:
            f.write(f"Running command:\n{' '.join(cmd)}\n\n")
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"✅ Completed {dataset}/{modality}/{lc} (log: {log_file})")
        return f"[DONE] {dataset}/{modality}/{lc}"
    except subprocess.CalledProcessError:
        print(f"❌ Failed {dataset}/{modality}/{lc} (log: {log_file})")
        return f"[FAILED] {dataset}/{modality}/{lc}"
    
def run_hpc(pair, args):
    dataset, modality, lc = pair["dataset"], pair["modality"], pair["lc"]
    sw_path, og_path = pair["sw_path"], pair["og_path"]
    save_dir = Path(args.save_base) / dataset / modality / lc
    save_dir.mkdir(parents=True, exist_ok=True)

    analyze_script = Path(args.project_dir) / "analysis" / "analyze_experiment.py"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    jobname = f"grad_{dataset}_{modality}_{lc}"
    sbatch_file = save_dir / f"sbatch_{jobname}.sh"

    cmd = f"{args.python_bin} {analyze_script} --dataset {dataset} --pred_sw {sw_path} --pred_og {og_path} --save_dir {save_dir} --tile_size 32 --bins 200 --channel 'all' --kl_start 29 --kl_end 33"
    if args.gradient_based_analysis:
        cmd += " --gradient_based_analysis"
    if args.qualitative_analysis:
        cmd += " --qualitative_analysis"
    if args.all:
        cmd += " --all"

    sbatch_contents = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={save_dir}/hpc/{jobname}.log
#SBATCH --error={save_dir}/hpc/{jobname}_err.log
#SBATCH --partition={args.partition}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --mem={args.mem}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --time={args.time}

cd {args.project_dir}
{cmd}
"""
    with open(sbatch_file, "w") as f:
        f.write(sbatch_contents)

    print(f"▶ Submitting HPC job for {dataset}/{modality}/{lc} via ssh hpc (login shell)")
    if not args.dry_run:
        # Use bash login shell so sbatch is in PATH
        ssh_cmd = f"ssh hpc \"bash -l -c 'sbatch {sbatch_file}'\""
        subprocess.run(ssh_cmd, shell=True, check=True)

    # Return string to append to main log file
    return f"[SUBMITTED {timestamp}] {dataset}/{modality}/{lc} (SBATCH: {sbatch_file})"

def main():
    args = parse_args()
    results_base = Path(args.results_base)
    save_base = Path(args.save_base)
    save_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_base / f"run_log_{timestamp}.log"

    print(f"🔍 Scanning for prediction pairs in {results_base}")
    pairs = find_prediction_pairs(results_base)
    print(f"Found {len(pairs)} valid SW–OG pairs.")
    if not pairs: return

    selected_pairs = select_pairs_interactively(pairs)
    results = []

    if args.hpc:
        for p in selected_pairs:
            results.append(run_hpc(p, args))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(run_local, p, args) for p in selected_pairs]
            for f in as_completed(futures):
                results.append(f.result())

    # Write all results (local + HPC submissions) to same log file
    with open(log_file, "w") as f:
        f.write("\n".join(results))

    print("\nSummary:")
    for r in results:
        print(" ", r)

    print(f"\n📝 Log saved to {log_file}")

if __name__ == "__main__":
    main()
