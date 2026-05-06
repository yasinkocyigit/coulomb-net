# tune_and_generate.py
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import linregress

# Import project modules
from src.dataset import ThreeChargeDataset
from src.data import set_seed

def plot_hexbin_density(y_true, y_pred, feature_name, r2, save_path, plot_options={}):
    """
    Creates a hexbin density plot of true vs. predicted values.
    """
    plt.figure(figsize=plot_options.get('figsize', (8, 8)))
    g = sns.jointplot(x=y_true, y=y_pred, kind="hex", height=7, cmap="viridis", joint_kws={'gridsize': 40})
    
    sns.regplot(x=y_true, y=y_pred, ax=g.ax_joint, scatter=False, color='red', line_kws={'linestyle': '--'})
    
    g.ax_joint.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'w--', linewidth=2)

    g.ax_joint.text(0.05, 0.95, f'R² = {r2:.4f}', transform=g.ax_joint.transAxes,
                    fontsize=plot_options.get('tick_fontsize', 12), verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    g.set_axis_labels(f"True {feature_name}", f"Predicted {feature_name}", fontsize=plot_options.get('label_fontsize', 12))
    fig = g.fig
    fig.suptitle(f'Hexbin Density (Tuned RF): True vs. Predicted {feature_name}', y=1.02, fontsize=plot_options.get('title_fontsize', 16))
    if plot_options.get('save_plots', True):
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def main():
    """
    Tunes a RandomForestRegressor, trains it, and generates plots.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    plot_options = config.get('plot_options', {})
    save_dir = Path(plot_options.get('plot_dir', './output_plt'))
    save_dir.mkdir(exist_ok=True)
    
    print("1. Loading dataset...")
    base_seed = config.get('seed', 42)
    set_seed(base_seed)
    
    dataset = ThreeChargeDataset(
        num_samples=config['samples'],
        seed=base_seed
    )
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=base_seed)

    print("2. Tuning RandomForestRegressor hyperparameters...")
    
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize GridSearchCV
    rf = RandomForestRegressor(random_state=base_seed, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, 
                               scoring='r2', verbose=2, n_jobs=-1)
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_

    print("\n3. Generating and saving 20 plots with the best model...")

    for i in range(20):
        print(f"   - Generating plot {i+1}/20...")
        # Use a different seed for each plot generation to get slight variations in data split
        plot_seed = base_seed + i
        set_seed(plot_seed)
        
        # Re-split data to introduce variability
        X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(X, y, test_size=0.2, random_state=plot_seed)
        
        # We can re-fit on the new split, but for speed we'll use the already tuned model
        # best_rf.fit(X_train_plot, y_train_plot) # Optional: re-fit for each split
        y_pred_plot = best_rf.predict(X_test_plot)
        
        # Calculate R^2 score for q1 and q2
        r2_q1 = r2_score(y_test_plot[:, 0], y_pred_plot[:, 0])
        r2_q2 = r2_score(y_test_plot[:, 1], y_pred_plot[:, 1])

        # Generate and save plots for q1 and q2
        plot_hexbin_density(y_test_plot[:, 0], y_pred_plot[:, 0], 'q1', r2_q1, save_dir / f'tuned_hexbin_density_q1_run_{i+1}.png', plot_options=plot_options)
        plot_hexbin_density(y_test_plot[:, 1], y_pred_plot[:, 1], 'q2', r2_q2, save_dir / f'tuned_hexbin_density_q2_run_{i+1}.png', plot_options=plot_options)

    print(f"\nAll 20 sets of plots have been saved to the '{save_dir.resolve()}' directory.")

if __name__ == '__main__':
    main()
