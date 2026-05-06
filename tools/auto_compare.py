#!/usr/bin/env python3
# auto_compare.py — patched to avoid pandas FutureWarning; supports --fast and forwards it to baseline_compare.py

import argparse, sys, shutil, json, glob
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Automated baseline comparison + tuning + figure collection")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--models", type=str, default="linear,rf,gb")
    p.add_argument("--tune_models", type=str, default="")
    p.add_argument("--n_iter", type=int, default=40)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--fast", action="store_true", help="Pass --fast to baseline_compare.py to trim heavy plots")
    return p.parse_args()

def run_cmd(cmd):
    import subprocess
    print("[auto_compare] Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode

def make_charts(df_all, figs_dir: Path):
    figs_dir.mkdir(parents=True, exist_ok=True)
    df_all["label"] = df_all["model"].str.upper() + " - " + df_all["variant"].str.capitalize()

    # R2(mean)
    fig = plt.figure(figsize=(8,4))
    order = df_all.sort_values("r2_mean", ascending=True)["label"]
    plt.barh(df_all.set_index("label").loc[order]["r2_mean"].index,
             df_all.set_index("label").loc[order]["r2_mean"].values)
    plt.xlabel("R² (mean across q1, q2)"); plt.title("Model Comparison: R² (mean)")
    plt.tight_layout(); fig.savefig(figs_dir / "comparison_r2_mean.png", dpi=160); plt.close(fig)

    # MSE(mean) (lower=better)
    fig = plt.figure(figsize=(8,4))
    order_mse = df_all.sort_values("mse_mean", ascending=False)["label"]
    plt.barh(df_all.set_index("label").loc[order_mse]["mse_mean"].index,
             df_all.set_index("label").loc[order_mse]["mse_mean"].values)
    plt.xlabel("MSE (mean across q1, q2)"); plt.title("Model Comparison: MSE (mean)")
    plt.tight_layout(); fig.savefig(figs_dir / "comparison_mse_mean.png", dpi=160); plt.close(fig)

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"auto_compare_{timestamp}"

    root = Path(args.outdir); run_dir = root / run_name
    figs_dir = run_dir / "figures"; tables_dir = run_dir / "tables"
    baselines_dir = run_dir / "baselines"; tuning_dir = run_dir / "tuning"
    for d in (figs_dir, tables_dir, baselines_dir, tuning_dir): d.mkdir(parents=True, exist_ok=True)

    # 1) baselines
    base_cmd = [sys.executable, "baseline_compare.py",
                "--config", args.config,
                "--outdir", str(baselines_dir),
                "--models", args.models,
                "--cv", str(args.cv)]
    if args.fast:
        base_cmd.append("--fast")
    rc = run_cmd(base_cmd)
    if rc != 0: print("[auto_compare][WARN] baselines step exited with code", rc)

    # 2) tuning (optional)
    if args.tune_models:
        rc = run_cmd([sys.executable, "baseline_tune.py",
                      "--config", args.config,
                      "--outdir", str(tuning_dir),
                      "--models", args.tune_models,
                      "--n_iter", str(args.n_iter),
                      "--cv", str(args.cv)])
        if rc != 0: print("[auto_compare][WARN] tuning step exited with code", rc)

    # 3) collect artifacts
    for p in baselines_dir.rglob("*.png"): shutil.copy2(p, figs_dir / p.name)
    for p in tuning_dir.rglob("*.png"): shutil.copy2(p, figs_dir / p.name)
    for ext in ("*.csv", "*.json"):
        for p in baselines_dir.rglob(ext): shutil.copy2(p, tables_dir / p.name)
        for p in tuning_dir.rglob(ext): shutil.copy2(p, tables_dir / p.name)

    # 4) combined summary (avoid concat with empty frame to prevent FutureWarning)
    def latest(pattern):
        files = sorted(glob.glob(str(tables_dir / pattern)))
        return files[-1] if files else None

    base_csv = latest("baseline_results_*.csv")
    tune_json = latest("tune_summary_*.json")

    common_cols = ["model", "variant", "r2_q1", "r2_q2", "r2_mean", "mse_q1", "mse_q2", "mse_mean"]
    frames = []

    if base_csv:
        dfb = pd.read_csv(base_csv); dfb["variant"] = "baseline"
        for col in common_cols:
            if col not in dfb.columns:
                dfb[col] = np.nan
        frames.append(dfb[common_cols])

    if tune_json:
        with open(tune_json, "r") as f:
            data = json.load(f)
        rows = [{
            "model": r["model"], "variant": "tuned",
            "r2_q1": r["test_r2_q1"], "r2_q2": r["test_r2_q2"], "r2_mean": r["test_r2_mean"],
            "mse_q1": r["test_mse_q1"], "mse_q2": r["test_mse_q2"], "mse_mean": r["test_mse_mean"],
        } for r in data]
        dft = pd.DataFrame(rows)
        frames.append(dft[common_cols])

    if frames:
        df_all = pd.concat(frames, ignore_index=True, sort=False)
        df_all.to_csv(tables_dir / "summary_combined.csv", index=False)
        make_charts(df_all, figs_dir)
        print("[auto_compare] Wrote summary_combined.csv and comparison charts.")
    else:
        print("[auto_compare] No results to summarize. (Did baseline/tuning produce outputs?)")

    print("All artifacts consolidated under:", run_dir.resolve())
    print("Figures:", figs_dir.resolve()); print("Tables:", tables_dir.resolve())

if __name__ == "__main__":
    main()
