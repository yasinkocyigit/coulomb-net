# generate_master_gallery.py
import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import shap

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

# --- Helper Functions & Classes ---

class TaylorDiagram:
    """Taylor Diagram for comparing model performance against a reference."""
    def __init__(self, ref_std, fig=None, rect=111, label='Reference'):
        self.ref_std = ref_std
        self.fig = fig if fig is not None else plt.figure()
        self.ax = self.fig.add_subplot(rect, polar=True, label='taylor')
        self.ax.set_thetalim(0, np.pi / 2)
        self.ax.set_rlim(0, 1.5 * self.ref_std)
        self.ax.set_xlabel("Standard Deviation")
        self.ax.set_ylabel("Correlation")
        self.ax.plot([0], self.ref_std, 'k*', ms=10, label=label)

    def add_sample(self, std, corr, label):
        self.ax.plot(np.arccos(corr), std, 'o', label=label)

    def add_grid(self, *args, **kwargs):
        self.ax.grid(*args, **kwargs)

# --- Plotting Functions ---

def plot_pareto_frontier(r2_scores, times, model_names, save_path):
    """Plots a Pareto frontier of R^2 vs. inference time."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(times, r2_scores, c='blue', s=50, alpha=0.7)
    for i, name in enumerate(model_names):
        ax.text(times[i] + 0.01 * np.mean(times), r2_scores[i], name, fontsize=9)
    
    # Sort by time to find frontier
    sorted_indices = np.argsort(times)
    pareto_front = []
    max_r2 = -np.inf
    for i in sorted_indices:
        if r2_scores[i] > max_r2:
            max_r2 = r2_scores[i]
            pareto_front.append((times[i], r2_scores[i]))
    pareto_front = np.array(pareto_front)
    ax.plot(pareto_front[:, 0], pareto_front[:, 1], 'r--', label='Pareto Frontier')

    ax.set_xlabel("Inference Time (seconds per sample)")
    ax.set_ylabel("R^2 Score")
    ax.set_title("Model Performance vs. Inference Time (Pareto Frontier)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_residual_violins(residuals_dict, save_path):
    """Plots violin plots of residuals for different models."""
    data_to_plot = []
    for model, residuals in residuals_dict.items():
        for i in range(residuals.shape[1]):
            for res in residuals[:, i]:
                data_to_plot.append({'Model': model, 'Output': f'q{i+1}', 'Residual': res})
    df = pd.DataFrame(data_to_plot)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(x='Model', y='Residual', hue='Output', data=df, split=True, inner='quart', ax=ax)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_title("Distribution of Model Residuals")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_pca_biplot(X, feature_names, save_path):
    """Performs PCA and creates a biplot of scores and loadings."""
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.2)
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.9, head_width=0.02)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='g', ha='center', va='center')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title("PCA Biplot of Input Features")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_correlation_dendrogram(X, feature_names, save_path):
    """Plots a dendrogram of feature correlations."""
    corr_matrix = np.corrcoef(X, rowvar=False)
    corr_linkage = linkage(corr_matrix, method='ward')
    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(corr_linkage, labels=feature_names, orientation='top', leaf_rotation=90)
    ax.set_title("Dendrogram of Feature Correlations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_e_field_quiver(X, feature_names, save_path):
    """Plots a 2D quiver plot of the E-field vectors."""
    r_target_x_idx = feature_names.index('r_target_x')
    r_target_y_idx = feature_names.index('r_target_y')
    e_x_idx = feature_names.index('E_total_x')
    e_y_idx = feature_names.index('E_total_y')

    # Subsample for clarity
    sample_indices = np.random.choice(X.shape[0], 500, replace=False)
    X_sample = X[sample_indices]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(X_sample[:, r_target_x_idx], X_sample[:, r_target_y_idx], 
              X_sample[:, e_x_idx], X_sample[:, e_y_idx], 
              color='blue', alpha=0.6, scale=5e9)
    ax.set_xlabel("r_target_x")
    ax.set_ylabel("r_target_y")
    ax.set_title("Electric Field Vector Map at Target Positions")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_shap_waterfall(model, X, feature_names, save_path):
    """Plots a SHAP waterfall chart for a single prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # We need to explain two outputs, so we do it for the first one (q1)
    # For a single prediction (the first sample)
    plt.figure()
    shap.waterfall_plot(shap.Explanation(values=shap_values[0][0], 
                                         base_values=explainer.expected_value[0], 
                                         data=X[0], feature_names=feature_names), show=False)
    plt.title("SHAP Waterfall for a Single Prediction (q1)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_lorenz_curve(model_errors, model_names, save_path):
    """Plots the Lorenz curve for model errors."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, errors in model_errors.items():
        abs_errors = np.abs(errors.mean(axis=1))
        sorted_errors = np.sort(abs_errors)
        cum_errors = np.cumsum(sorted_errors)
        total_error = cum_errors[-1]
        lorenz_curve = cum_errors / total_error
        
        n_samples = len(errors)
        percent_samples = np.linspace(0., 1., n_samples)
        
        gini = (0.5 - np.trapz(lorenz_curve, percent_samples)) / 0.5
        ax.plot(percent_samples, lorenz_curve, label=f'{name} (Gini={gini:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Line of Equality')
    ax.set_xlabel("Cumulative Share of Samples")
    ax.set_ylabel("Cumulative Share of Absolute Error")
    ax.set_title("Lorenz Curve of Model Errors")
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate a Master Gallery of Plots")
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
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    feature_names = ThreeChargeDataset.feature_names()

    print("2. Training models (RF and XGB)...")
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    models = {"RandomForest": rf, "XGBoost": xgb}
    
    model_errors = {}
    model_r2 = {}
    model_times = {}

    for name, model in models.items():
        print(f"   - Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)

        model_errors[name] = y_test - y_pred
        model_r2[name] = model.score(X_test, y_test)
        model_times[name] = inference_time

    print("3. Generating master plot gallery...")

    # Pareto Frontier
    print("   - Plotting Pareto Frontier...")
    plot_pareto_frontier(list(model_r2.values()), list(model_times.values()), list(models.keys()), save_dir / "pareto_frontier.png")

    # Taylor Diagram
    print("   - Plotting Taylor Diagram...")
    ref_std_q1 = np.std(y_test[:, 0])
    taylor_diag = TaylorDiagram(ref_std_q1, label='Reference (q1)')
    for name, model in models.items():
        y_pred = model.predict(X_test)
        corr = np.corrcoef(y_test[:, 0], y_pred[:, 0])[0, 1]
        std = np.std(y_pred[:, 0])
        taylor_diag.add_sample(std, corr, label=name)
    taylor_diag.add_grid()
    taylor_diag.fig.legend(loc='upper right')
    taylor_diag.fig.savefig(save_dir / "taylor_diagram_q1.png", dpi=150)
    plt.close(taylor_diag.fig)

    # Violin Plots
    print("   - Plotting Violin Plots of Residuals...")
    plot_residual_violins(model_errors, save_dir / "residual_violins.png")

    # PCA Biplot
    print("   - Plotting PCA Biplot...")
    plot_pca_biplot(X, feature_names, save_dir / "pca_biplot.png")

    # Dendrogram
    print("   - Plotting Correlation Dendrogram...")
    plot_correlation_dendrogram(X, feature_names, save_dir / "correlation_dendrogram.png")

    # Quiver Plot
    print("   - Plotting E-Field Quiver Plot...")
    plot_e_field_quiver(X_test, feature_names, save_dir / "e_field_quiver.png")

    # SHAP Waterfall
    print("   - Plotting SHAP Waterfall...")
    plot_shap_waterfall(rf, X_test[:100], feature_names, save_dir / "shap_waterfall_rf.png")

    # Lorenz Curve
    print("   - Plotting Lorenz Curve...")
    plot_lorenz_curve(model_errors, list(models.keys()), save_dir / "lorenz_curve_errors.png")

    print(f"\nMaster gallery plots saved to '{save_dir.resolve()}'.")

if __name__ == '__main__':
    main()
