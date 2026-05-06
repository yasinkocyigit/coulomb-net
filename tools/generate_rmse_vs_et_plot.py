# generate_plots.py
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import ecdf, linregress

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

# --- Plotting Functions ---

def plot_learning_curve_with_ci(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve with confidence bands.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score (R^2)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="r2")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid(True, linestyle='--', alpha=0.6)
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    
    return plt

def plot_hexbin_density(y_true, y_pred, feature_name, save_path):
    """
    Creates a hexbin density plot of true vs. predicted values with a fitted trend line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Using jointplot for a clean hexbin with marginal distributions
    g = sns.jointplot(x=y_true, y=y_pred, kind="hex", height=7, cmap="viridis", joint_kws={'gridsize': 40})
    
    # Overlay regression line
    sns.regplot(x=y_true, y=y_pred, ax=g.ax_joint, scatter=False, color='red', line_kws={'linestyle': '--'})
    
    # Add y=x line for reference
    g.ax_joint.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'w--', linewidth=2)

    slope, intercept, r_value, _, _ = linregress(y_true, y_pred)
    g.ax_joint.text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=g.ax_joint.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    g.set_axis_labels(f"True {feature_name}", f"Predicted {feature_name}")
    fig = g.fig
    fig.suptitle(f'Hexbin Density: True vs. Predicted {feature_name}', y=1.02)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_radar_chart(data, categories, title, save_path):
    """
    Creates a radar chart for comparing multiple models across several metrics.
    """
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, scores in data.items():
        scores_closed = scores + scores[:1]
        ax.plot(angles, scores_closed, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, scores_closed, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)

    ax.set_rlabel_position(180 / num_vars)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="grey", size=10)
    ax.set_ylim(0, 1.05)

    ax.set_title(title, size=16, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_contour_error_map(model, X_test, y_test, feature_indices, feature_names, save_path):
    """
    Plots a contour map of the model's error as a function of two features.
    Note: This is an approximation, as it averages over other features.
    """
    f1_idx, f2_idx = feature_indices
    f1_name, f2_name = feature_names[f1_idx], feature_names[f2_idx]

    # Create a grid for the two features
    f1_range = np.linspace(X_test[:, f1_idx].min(), X_test[:, f1_idx].max(), 20)
    f2_range = np.linspace(X_test[:, f2_idx].min(), X_test[:, f2_idx].max(), 20)
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

    mse_grid = np.zeros_like(f1_grid)
    
    # Calculate MSE at each grid point
    for i in range(f1_grid.shape[0]):
        for j in range(f1_grid.shape[1]):
            # Create a sample by taking the mean of all other features
            sample = np.mean(X_test, axis=0)
            sample[f1_idx] = f1_grid[i, j]
            sample[f2_idx] = f2_grid[i, j]
            
            # Predict and find a plausible true y to calculate error
            # This is a simplification: we find the nearest neighbor in the test set
            # to estimate a "local" true value.
            distances = np.linalg.norm(X_test[:, [f1_idx, f2_idx]] - np.array([sample[f1_idx], sample[f2_idx]]), axis=1)
            nearest_neighbor_idx = np.argmin(distances)
            
            y_true_local = y_test[nearest_neighbor_idx]
            y_pred_local = model.predict(sample.reshape(1, -1))[0]
            
            mse_grid[i, j] = np.mean((y_true_local - y_pred_local)**2)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(f1_grid, f2_grid, mse_grid, levels=15, cmap='magma')
    cbar = fig.colorbar(contour)
    cbar.set_label('Mean Squared Error (MSE)')
    
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    ax.set_title(f'Model Error Contour Map vs {f1_name} and {f2_name}')
    
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_ecdf_residuals(y_true, y_pred, model_name, save_path):
    """
    Plots the ECDF of the model's residuals with a 95% confidence interval.
    """
    residuals_q1 = y_true[:, 0] - y_pred[:, 0]
    residuals_q2 = y_true[:, 1] - y_pred[:, 1]

    # Process q1
    res_q1_ecdf = ecdf(residuals_q1)
    quantiles_q1 = res_q1_ecdf.cdf.quantiles
    probabilities_q1 = res_q1_ecdf.cdf.probabilities
    ci_q1 = res_q1_ecdf.cdf.confidence_interval()

    # Interpolate the CI ECDF values onto the main ECDF's quantiles
    q1_low_probs = np.interp(quantiles_q1, ci_q1.low.quantiles, ci_q1.low.probabilities)
    q1_high_probs = np.interp(quantiles_q1, ci_q1.high.quantiles, ci_q1.high.probabilities)

    # Process q2
    res_q2_ecdf = ecdf(residuals_q2)
    quantiles_q2 = res_q2_ecdf.cdf.quantiles
    probabilities_q2 = res_q2_ecdf.cdf.probabilities
    ci_q2 = res_q2_ecdf.cdf.confidence_interval()
    
    # Interpolate the CI ECDF values onto the main ECDF's quantiles
    q2_low_probs = np.interp(quantiles_q2, ci_q2.low.quantiles, ci_q2.low.probabilities)
    q2_high_probs = np.interp(quantiles_q2, ci_q2.high.quantiles, ci_q2.high.probabilities)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for q1
    ax1.step(quantiles_q1, probabilities_q1, where='post', label='ECDF of Residuals (q1)')
    ax1.fill_between(quantiles_q1, q1_low_probs, q1_high_probs, alpha=0.2, label='95% CI')
    ax1.axvline(0, color='r', linestyle='--', label='Zero Error')
    ax1.set_title(f'ECDF of Residuals for q1 ({model_name})')
    ax1.set_xlabel('Residual (True - Predicted)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot for q2
    ax2.step(quantiles_q2, probabilities_q2, where='post', label='ECDF of Residuals (q2)')
    ax2.fill_between(quantiles_q2, q2_low_probs, q2_high_probs, alpha=0.2, label='95% CI')
    ax2.axvline(0, color='r', linestyle='--', label='Zero Error')
    ax2.set_title(f'ECDF of Residuals for q2 ({model_name})')
    ax2.set_xlabel('Residual (True - Predicted)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

def plot_rmse_vs_feature_with_min_annotation(y_true, y_pred, feature_values, feature_name, save_path, n_bins=50):
    """
    Plots RMSE against a feature, finds the minimum RMSE, and annotates it on the plot.
    """
    squared_errors = np.sum((y_true - y_pred)**2, axis=1)
    df = pd.DataFrame({'feature': feature_values, 'se': squared_errors})
    
    # Ensure there are enough unique values to create bins
    if df['feature'].nunique() < n_bins:
        n_bins = df['feature'].nunique()

    df['feature_bin'] = pd.cut(df['feature'], bins=n_bins, duplicates='drop')
    binned_rmse = df.groupby('feature_bin')['se'].mean().apply(np.sqrt)

    bin_midpoints = [interval.mid for interval in binned_rmse.index]
    
    # Find the minimum point
    min_rmse_val = binned_rmse.min()
    min_bin_idx = binned_rmse.idxmin()
    min_feature_val = min_bin_idx.mid

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(bin_midpoints, binned_rmse.values, marker='o', linestyle='-', color='dodgerblue', label='Binned RMSE')

    # Annotate the minimum point
    ax.plot(min_feature_val, min_rmse_val, '*', color='red', markersize=15, label=f'En Düşük Hata (RMSE: {min_rmse_val:.3f})')
    ax.annotate(
        f'Minimum Noktası\n{feature_name} ≈ {min_feature_val:.2e}\nRMSE ≈ {min_rmse_val:.3f}',
        xy=(min_feature_val, min_rmse_val),
        xytext=(min_feature_val, min_rmse_val + (binned_rmse.max() - min_rmse_val) * 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6)
    )

    ax.set_xlabel(f'{feature_name}')
    ax.set_ylabel('Kök Ortalama Karesel Hata (RMSE)')
    ax.set_title(f'Model Hatasının {feature_name} Değerine Göre Değişimi')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate RMSE vs. E_total Plot with Minimum Annotation")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = Path('./output_plt')
    save_dir.mkdir(exist_ok=True)
    
    print("1. Loading dataset...")
    seed = config.get('seed', 42)
    set_seed(seed)
    dataset = ThreeChargeDataset(num_samples=config['samples'], seed=seed)
    
    scaler = StandardScaler().fit(dataset.X)
    X_scaled = scaler.transform(dataset.X)
    X_train, X_test, y_train, y_test, X_orig_train, X_orig_test = train_test_split(
        X_scaled, dataset.y, dataset.X, test_size=0.2, random_state=seed
    )

    print("2. Training RandomForest model...")
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("3. Generating and saving annotated plot...")

    feature_names_list = ThreeChargeDataset.feature_names()
    E_total_x_idx = feature_names_list.index('E_total_x')
    E_total_y_idx = feature_names_list.index('E_total_y')
    E_total_z_idx = feature_names_list.index('E_total_z')
    
    E_vec = X_orig_test[:, [E_total_x_idx, E_total_y_idx, E_total_z_idx]]
    E_mag = np.linalg.norm(E_vec, axis=1)

    # Generate the plot with 500 bins for higher granularity
    plot_rmse_vs_feature_with_min_annotation(
        y_test, 
        y_pred_rf, 
        E_mag, 
        'E_total Magnitude', 
        save_dir / 'rmse_vs_etotal_magnitude_annotated_500bins.png',
        n_bins=500
    )

    print(f"\nHigh-granularity annotated plot has been saved to the '{save_dir.resolve()}\rmse_vs_etotal_magnitude_annotated_500bins.png' directory.")

if __name__ == '__main__':
    main()
