# generate_etotal_error_plots.py
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

# --- Plotting Functions ---

def plot_error_vs_etotal_scatter(e_total, error, c, c_label, save_path):
    """
    Creates a scatter plot of error vs E_total, with points colored by a third variable.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(e_total, error, c=c, cmap='viridis', alpha=0.5)
    ax.set_xlabel('E_total')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Error vs E_total')
    ax.grid(True, linestyle='--', alpha=0.6)
    cbar = fig.colorbar(scatter)
    cbar.set_label(c_label)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_error_vs_etotal_hexbin(e_total, error, save_path):
    """
    Creates a hexbin plot of error vs E_total.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.jointplot(x=e_total, y=error, kind="hex", height=7, cmap="viridis", joint_kws={'gridsize': 40})
    g.set_axis_labels('E_total', 'Mean Squared Error (MSE)')
    fig = g.fig
    fig.suptitle('Error vs E_total (Hexbin Density)', y=1.02)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_binned_error_vs_etotal(e_total, error, save_path):
    """
    Creates a binned scatter plot of error vs E_total.
    """
    df = pd.DataFrame({'E_total': e_total, 'error': error})
    df['E_total_bin'] = pd.cut(df['E_total'], bins=20)
    binned_error = df.groupby('E_total_bin')['error'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    binned_error.plot(kind='bar', ax=ax)
    ax.set_xlabel('E_total Bins')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Binned Error vs E_total')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate E_total vs Error Plots for projectv10")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = Path('./output_plot3')
    save_dir.mkdir(exist_ok=True)
    
    print("1. Loading dataset...")
    seed = config.get('seed', 42)
    set_seed(seed)
    dataset = ThreeChargeDataset(
        num_samples=config['samples'],
        seed=seed
    )
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print("2. Training RandomForest model...")
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("3. Calculating E_total and error...")
    feature_names_list = ThreeChargeDataset.feature_names()
    E_total_x_idx = feature_names_list.index('E_total_x')
    E_total_y_idx = feature_names_list.index('E_total_y')
    E_total_z_idx = feature_names_list.index('E_total_z')
    E_vec = X_test[:, [E_total_x_idx, E_total_y_idx, E_total_z_idx]]
    E_mag = np.linalg.norm(E_vec, axis=1)

    mse_per_sample = np.mean((y_test - y_pred_rf)**2, axis=1)

    print("4. Generating and saving plots...")

    # Scatter plot colored by q1
    plot_error_vs_etotal_scatter(E_mag, mse_per_sample, y_test[:, 0], 'q1', save_dir / 'error_vs_etotal_scatter_q1.png')
    print("   - Saved error_vs_etotal_scatter_q1.png")

    # Scatter plot colored by q2
    plot_error_vs_etotal_scatter(E_mag, mse_per_sample, y_test[:, 1], 'q2', save_dir / 'error_vs_etotal_scatter_q2.png')
    print("   - Saved error_vs_etotal_scatter_q2.png")

    # Hexbin plot
    plot_error_vs_etotal_hexbin(E_mag, mse_per_sample, save_dir / 'error_vs_etotal_hexbin.png')
    print("   - Saved error_vs_etotal_hexbin.png")

    # Binned plot
    plot_binned_error_vs_etotal(E_mag, mse_per_sample, save_dir / 'binned_error_vs_etotal.png')
    print("   - Saved binned_error_vs_etotal.png")

    print(f"\nAll plots have been saved to the '{save_dir.resolve()}' directory.")

if __name__ == '__main__':
    main()
