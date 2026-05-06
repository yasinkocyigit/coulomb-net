import yaml
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

def parse_metrics_from_output(output):
    mse_q1 = mse_q2 = r2_q1 = r2_q2 = None
    mse_match = re.search(r"Test MSE for q1: (\d+\.\d+), q2: (\d+\.\d+)", output)
    r2_match = re.search(r"Test R\^2\s+for q1: (\d+\.\d+), q2: (\d+\.\d+)", output)

    if mse_match:
        mse_q1 = float(mse_match.group(1))
        mse_q2 = float(mse_match.group(2))
    if r2_match:
        r2_q1 = float(r2_match.group(1))
        r2_q2 = float(r2_match.group(2))
    
    return {"mse_q1": mse_q1, "mse_q2": mse_q2, "r2_q1": r2_q1, "r2_q2": r2_q2}

def main():
    project_root = Path("/Users/tarikak/three_charge_predictor/projectv11/projectv11")
    config_path = project_root / "config.yaml"
    main_script_path = project_root / "main.py"
    python_executable = project_root / "venv/bin/python"

    # Read original config content
    with open(config_path, 'r') as f:
        original_config_content = f.read()

    f_min_values = np.logspace(5, 12, 20) # 1e5 to 1e12, 20 points

    results = []
    for f_min in f_min_values:
        print(f"Running experiment for f_min = {f_min:.2e}")
        
        # Modify config.yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['f_min'] = float(f_min) # Ensure it's a float for YAML
        config['e_min'] = None # Set e_min to None to isolate f_min effect
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run main.py
        command = [
            str(python_executable),
            str(main_script_path),
            "--config",
            str(config_path),
            "--early_stopping" # Enable early stopping for faster runs
        ]
        
        process = subprocess.run(command, capture_output=True, text=True, cwd=project_root)
        
        # Parse output
        metrics = parse_metrics_from_output(process.stdout)
        metrics["f_min"] = f_min
        results.append(metrics)
        
        # Revert config.yaml
        with open(config_path, 'w') as f:
            f.write(original_config_content)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    f_mins_plot = [r['f_min'] for r in results]
    mse_q1 = [r['mse_q1'] for r in results]
    mse_q2 = [r['mse_q2'] for r in results]
    r2_q1 = [r['r2_q1'] for r in results]
    r2_q2 = [r['r2_q2'] for r in results]

    ax1.plot(f_mins_plot, mse_q1, 'o-', label='MSE q1')
    ax1.plot(f_mins_plot, mse_q2, 'o-', label='MSE q2')
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.set_title("Prediction Accuracy vs. Minimum Force Threshold")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(f_mins_plot, r2_q1, 'o-', label='R^2 q1')
    ax2.plot(f_mins_plot, r2_q2, 'o-', label='R^2 q2')
    ax2.set_xlabel("Minimum Force (|F_total|) Threshold (N)")
    ax2.set_ylabel("R-squared (R^2)")
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(project_root / "fmin_vs_accuracy.png")
    print(f"Plot saved to {project_root / 'fmin_vs_accuracy.png'}")

if __name__ == "__main__":
    main()
