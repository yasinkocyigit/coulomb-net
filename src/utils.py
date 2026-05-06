# src/utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_loss(history, save_path):
    plt.figure()
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(12, 5))
    for i, label in enumerate(['q1', 'q2']):
        plt.subplot(1, 2, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.4)
        plt.plot([-5, 5], [-5, 5], 'r--')
        plt.title(f'{label}: Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 5))
    for i, label in enumerate(['q1', 'q2']):
        plt.subplot(1, 2, i+1)
        plt.hist(residuals[:, i], bins=50, alpha=0.7, color='gray', edgecolor='black')
        plt.title(f'Residual Histogram for {label}')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_distribution(y_true, y_pred, save_path):
    plt.figure(figsize=(12, 5))
    for i, label in enumerate(['q1', 'q2']):
        plt.subplot(1, 2, i+1)
        sns.kdeplot(y_true[:, i], label='True', fill=True, linewidth=2)
        sns.kdeplot(y_pred[:, i], label='Predicted', fill=True, linewidth=2)
        plt.title(f'Distribution of {label}: True vs Predicted')
        plt.xlabel(label)
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_vs_charge_magnitude(y_true, y_pred, save_path):
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(12, 5))
    for i, label in enumerate(['q1', 'q2']):
        plt.subplot(1, 2, i+1)
        plt.scatter(np.abs(y_true[:, i]), errors[:, i], alpha=0.5)
        plt.xlabel(f'|True {label}|')
        plt.ylabel(f'|Error in {label}|')
        plt.title(f'Error vs. Charge Magnitude for {label}')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === EXTRA: metrics IO helpers (save_metrics, ensure_dir) ===
from pathlib import Path
def ensure_dir(path: str) -> None:
    """
    Ensure that the directory for the given path exists. If the path has a file suffix,
    ensure the parent directory exists; otherwise, create the directory itself.
    """
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

def save_metrics(y_true, y_pred, out_txt_path: str, extra=None):
    """
    Compute basic regression metrics (MSE, R^2, MAE) and save them to a plain text file.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Predicted target values.
    out_txt_path : str
        Path to the output .txt file.
    extra : dict, optional
        Additional metrics to include in the file. Keys are metric names and values are numeric.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    ensure_dir(out_txt_path)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    lines = []
    lines.append("Regression metrics:")
    lines.append(f"  MSE = {mse:.6f}")
    lines.append(f"  R2  = {r2:.6f}")
    lines.append(f"  MAE = {mae:.6f}")
    if extra:
        for k, v in extra.items():
            try:
                lines.append(f"  {k} = {float(v):.6f}")
            except Exception:
                lines.append(f"  {k} = {v}")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"MSE": mse, "R2": r2, "MAE": mae}
