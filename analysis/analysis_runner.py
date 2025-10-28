#!/usr/bin/env python3
"""
Batch Experiment Analysis Runner with HPC support
-------------------------------------------------
Supports 'usplit' and 'microsplit' folder structures.
Can run locally or submit to HPC via SLURM.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer

app = typer.Typer(help="Run batch experiment analysis locally or on HPC")

# -------------------------------------------------------------------
# 🔹 Core helpers
# -------------------------------------------------------------------
def find_prediction_pairs(results_base: Path, model_name: str):
    """Find SW–OG prediction pairs given folder structure."""
    pairs = []

    if model_name == "usplit":
        for dataset_dir in results_base.iterdir():
            if not dataset_dir.is_dir(): continue
            for modality_dir in dataset_dir.iterdir():
                if not modality_dir.is_dir(): continue
                for lc_dir in modality_dir.iterdir():
                    if not lc_dir.is_dir(): continue
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
    elif model_name == "microsplit":
        for dataset_dir in results_base.iterdir():
            if not dataset_dir.is_dir(): continue
            og_files = list(dataset_dir.glob("*_og.pkl*"))
            sw_files = list(dataset_dir.glob("*_stitched.pkl*"))
            if og_files and sw_files:
                pairs.append({
                    "dataset": dataset_dir.name,
                    "og_path": str(og_files[0]),
                    "sw_path": str(sw_files[0]),
                })

    return pairs

def build_analysis_cmd(args, pair):
    """Builds the python command list for analysis."""
    analyze_script = Path(args["project_dir"]) / "analysis" / "analyze_experiment.py"
    cmd = [
        args["python_bin"], str(analyze_script),
        "--model_name", args["model_name"],
        "--dataset", pair["dataset"],
        "--pred_sw", pair["sw_path"],
        "--pred_og", pair["og_path"],
        "--save_dir", str(Path(args["save_base"]) / pair["dataset"]),
        "--inner_tile_size", "32",
        "--bins", "200",
        "--channel", "all",
        "--kl_start", "29",
        "--kl_end", "33",
    ]
    if args["gradient_based_analysis"]:
        cmd.append("--gradient_based_analysis")
    if args["qualitative_analysis"]:
        cmd.append("--qualitative_analysis")
    if args["all"]:
        cmd.append("--all")
    return cmd

# -------------------------------------------------------------------
# 🔹 Execution
# -------------------------------------------------------------------
def run_local(pair, args):
    save_dir = Path(args["save_base"]) / pair["dataset"]
    if "modality" in pair: save_dir /= pair["modality"]
    if "lc" in pair: save_dir /= pair["lc"]
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_analysis_cmd(args, pair)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_dir / f"analysis_log_{timestamp}.log"

    if args["dry_run"]:
        print("[DRY RUN]", " ".join(cmd))
        return f"[SKIPPED] {pair}"

    print(f"▶ Running local analysis for {pair}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"✅ Completed {pair}")
    return f"[DONE] {pair}"

def run_hpc(pair, args):
    """Writes and submits SLURM job."""
    save_dir = Path(args["save_base"]) / pair["dataset"]
    if "modality" in pair: save_dir /= pair["modality"]
    if "lc" in pair: save_dir /= pair["lc"]
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd_str = " ".join(build_analysis_cmd(args, pair))
    jobname = f"grad_{pair['dataset']}"
    sbatch_file = save_dir / f"sbatch_{jobname}.sh"

    sbatch_contents = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={save_dir}/hpc_{jobname}.log
#SBATCH --error={save_dir}/hpc_{jobname}_err.log
#SBATCH --partition={args['partition']}
#SBATCH --gres=gpu:{args['gpus']}
#SBATCH --mem={args['mem']}
#SBATCH --cpus-per-task={args['cpus']}
#SBATCH --time={args['time']}

cd {args['project_dir']}
{cmd_str}
"""
    sbatch_file.write_text(sbatch_contents)

    if not args["dry_run"]:
        subprocess.run(f"ssh hpc 'bash -l -c \"sbatch {sbatch_file}\"'", shell=True, check=True)

    return f"[SUBMITTED] {pair} ({sbatch_file})"

# -------------------------------------------------------------------
# 🔹 CLI entry point
# -------------------------------------------------------------------
@app.command()
def main(
    model_name: str = typer.Option(..., help="usplit or microsplit"),
    results_base: Path = typer.Option("/group/jug/aman/usplit_13Oct25/"),
    project_dir: Path = typer.Option("/home/aman.kukde/sliding_windowed_tiling/"),
    save_base: Path = typer.Option("/group/jug/aman/usplit_analysis_results_16Oct25/"),
    python_bin: Path = typer.Option("/scratch/aman.kukde/conda/envs/msr/bin/python3.10"),
    max_workers: int = typer.Option(4),
    dry_run: bool = typer.Option(False),
    gradient_based_analysis: bool = typer.Option(False),
    qualitative_analysis: bool = typer.Option(False),
    all: bool = typer.Option(False),
    hpc: bool = typer.Option(False),
    partition: str = typer.Option("gpuq"),
    gpus: int = typer.Option(1),
    mem: str = typer.Option("64GB"),
    cpus: int = typer.Option(4),
    time: str = typer.Option("12:00:00"),
):
    args = locals()  # 👈 pass around as dict
    pairs = find_prediction_pairs(results_base, model_name)
    print(f"Found {len(pairs)} valid pairs.")
    if not pairs: raise typer.Exit()

    results = []
    if hpc:
        for p in pairs:
            results.append(run_hpc(p, args))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_local, p, args) for p in pairs]
            for f in as_completed(futures):
                results.append(f.result())

    log_dir = save_base / "analysis_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (log_dir / f"run_log_{timestamp}.log").write_text("\n".join(results))
    print("✅ All done.")

if __name__ == "__main__":
    app()
