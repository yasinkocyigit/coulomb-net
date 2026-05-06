# generate_contour_map.py
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Proje modüllerini import et
from src.dataset import ThreeChargeDataset
from src.data import set_seed

def main():
    parser = argparse.ArgumentParser(description="Generate Contour Error Map for E_total_mag vs F_total_mag")
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

    print("2. Training RandomForest model...")
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("3. Calculating error and magnitudes...")
    y_pred = rf.predict(X_test)
    mse_per_sample = np.mean((y_test - y_pred)**2, axis=1)

    feature_names = ThreeChargeDataset.feature_names()
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['E_total_mag'] = np.linalg.norm(df_test[['E_total_x', 'E_total_y', 'E_total_z']].values, axis=1)
    df_test['F_total_mag'] = np.linalg.norm(df_test[['F_total_x', 'F_total_y', 'F_total_z']].values, axis=1)
    df_test['mse'] = mse_per_sample

    print("4. Generating and saving contour error map...")
    
    plt.figure(figsize=(10, 8))
    # Using hexbin as a proxy for a contour map of error
    hb = plt.hexbin(df_test['E_total_mag'], df_test['F_total_mag'], C=df_test['mse'], gridsize=30, cmap='viridis', reduce_C_function=np.mean)
    
    cb = plt.colorbar(hb)
    cb.set_label('Mean Squared Error (MSE)')
    
    plt.xlabel('E_total_mag')
    plt.ylabel('F_total_mag')
    plt.title('Contour Error Map of E_total_mag vs F_total_mag')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = save_dir / 'contour_error_map_Emag_vs_Fmag.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nContour error map saved to '{save_path.resolve()}'")

if __name__ == '__main__':
    main()
