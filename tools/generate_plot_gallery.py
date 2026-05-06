# generate_plot_gallery.py
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Proje modüllerini ve önceki çizim fonksiyonlarını import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed
from src.extra_viz import plot_pdp_ice # PDP/ICE plot
from generate_plots import (
    plot_learning_curve_with_ci, 
    plot_contour_error_map,
)

def main():
    parser = argparse.ArgumentParser(description="Generate a Gallery of Advanced Plots for projectv10")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = Path('./output_plt')
    save_dir.mkdir(exist_ok=True)
    
    print("1. Loading dataset...")
    seed = config.get('seed', 42)
    set_seed(seed)
    dataset = ThreeChargeDataset(
        num_samples=config['samples'],
        seed=seed
    )
    X, y = dataset.X, dataset.y
    # Using a smaller subset for faster PDP generation
    X_pdp, _, y_pdp, _ = train_test_split(X, y, train_size=0.1, random_state=seed)

    print("2. Training models (RF and XGB)...")
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    
    # Train RF for general use
    rf.fit(X, y)

    feature_names = ThreeChargeDataset.feature_names()

    print("3. Generating plot gallery...")

    # --- Learning Curve Gallery ---
    print("   - Generating Learning Curves for RF and XGB...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    plot_learning_curve_with_ci(rf, "Learning Curve (Random Forest)", X, y, axes=axes[0], cv=3)
    plot_learning_curve_with_ci(xgb, "Learning Curve (XGBoost)", X, y, axes=axes[1], cv=3)
    fig.tight_layout()
    plt.savefig(save_dir / 'learning_curve_gallery.png', dpi=150)
    plt.close(fig)

    # --- Contour Map Gallery ---
    print("   - Generating Contour Maps for different feature sets...")
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)
    top_2_indices = sorted_indices[-2:]
    bottom_2_indices = sorted_indices[:2]

    # Re-split with a test set for error calculation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf.fit(X_train, y_train) # Re-fit on training data only

    plot_contour_error_map(rf, X_test, y_test, top_2_indices, feature_names, save_dir / 'contour_map_top_features.png')
    plot_contour_error_map(rf, X_test, y_test, bottom_2_indices, feature_names, save_dir / 'contour_map_bottom_features.png')
    print(f"   - Top 2 features: {[feature_names[i] for i in top_2_indices]}")
    print(f"   - Bottom 2 features: {[feature_names[i] for i in bottom_2_indices]}")

    # --- Partial Dependence Plot (PDP/ICE) Gallery ---
    print("   - Generating PDP/ICE plots for top 4 features...")
    top_4_indices = sorted_indices[-4:]
    
    for feature_idx in top_4_indices:
        feature_name = feature_names[feature_idx]
        safe_feature_name = feature_name.replace('/', '_') # Sanitize filename
        save_path = save_dir / f'pdp_ice_for_{safe_feature_name}.png'
        print(f"     - Plotting PDP for: {feature_name}")
        plot_pdp_ice(rf, X_pdp, feature_idx, save_path=save_path, feature_name=feature_name)
    
    print(f"   - PDPs generated for: {[feature_names[i] for i in top_4_indices]}")

    print(f"\nAll new plot gallery files have been saved to the '{save_dir.resolve()}' directory.")

if __name__ == '__main__':
    main()
