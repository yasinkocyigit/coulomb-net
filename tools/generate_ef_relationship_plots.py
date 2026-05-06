# generate_ef_relationship_plots.py
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

def main():
    parser = argparse.ArgumentParser(description="Generate E_total vs F_total relationship plots")
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

    feature_names = ThreeChargeDataset.feature_names()
    df = pd.DataFrame(X, columns=feature_names)

    # Calculate magnitudes
    df['E_total_mag'] = np.linalg.norm(df[['E_total_x', 'E_total_y', 'E_total_z']].values, axis=1)
    df['F_total_mag'] = np.linalg.norm(df[['F_total_x', 'F_total_y', 'F_total_z']].values, axis=1)

    plots_to_generate = [
        ('E_total_mag', 'F_total_mag'),
        ('E_total_x', 'F_total_x'),
        ('E_total_y', 'F_total_y'),
        ('E_total_z', 'F_total_z'),
        ('E_total_x', 'F_total_y'),
        ('E_total_x', 'F_total_z'),
        ('E_total_y', 'F_total_x'),
        ('E_total_y', 'F_total_z'),
        ('E_total_z', 'F_total_x'),
        ('E_total_z', 'F_total_y'),
    ]

    print(f"2. Generating and saving {len(plots_to_generate)} scatter plots...")

    for feature1, feature2 in plots_to_generate:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=feature1, y=feature2, alpha=0.5)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.grid(True)
        save_path = save_dir / f'{feature1}_vs_{feature2}_scatter.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  - Saved {save_path.name}")

    print(f"\n{len(plots_to_generate)} scatter plots saved to '{save_dir.resolve()}'")

if __name__ == '__main__':
    main()
