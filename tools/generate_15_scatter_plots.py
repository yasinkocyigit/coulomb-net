# generate_15_scatter_plots.py
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import itertools

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

def main():
    parser = argparse.ArgumentParser(description="Generate 15 Scatter Plots for projectv10")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = Path('./output_plt4')
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

    print("2. Training RandomForest model to get feature importances...")
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    feature_names = ThreeChargeDataset.feature_names()
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_10_indices = indices[:top_n]
    top_10_features = [feature_names[i] for i in top_10_indices]

    print("Top 10 most important features:")
    for i in range(top_n):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

    print("3. Generating and saving 15 scatter plots...")
    
    df = pd.DataFrame(X, columns=feature_names)

    plot_count = 0
    for feature1, feature2 in itertools.combinations(top_10_features, 2):
        if plot_count >= 15:
            break

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=feature1, y=feature2)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.grid(True)
        save_path = save_dir / f'{feature1}_vs_{feature2}_scatter.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  - Saved {save_path.name}")
        plot_count += 1

    print(f"\n{plot_count} scatter plots saved to '{save_dir.resolve()}'")

if __name__ == '__main__':
    main()
