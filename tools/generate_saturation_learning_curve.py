
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import learning_curve
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.dataset import ThreeChargeDataset
from src.data import set_seed

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate Saturation Learning Curves")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    config = load_config(args.config)

    # Setup directories and seed
    output_dir = Path("C:/Users/yasin/Documents/projectv10/projectv10/output_plt2")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.get('seed', 42)
    set_seed(seed)

    # Load a very large dataset to find the saturation point
    # We need 120k total samples to get 80k for training (80,000 * 3/2 = 120,000)
    print("Generating a very large dataset (120,000 samples). This may take a while...")
    dataset = ThreeChargeDataset(
        num_samples=120000, # Increased sample size to 120k
        position_range=config.get('position_range', 10.0),
        charge_range=config.get('charge_range', 5.0),
        seed=seed
    )
    X, y = dataset.X, dataset.y

    # Define the best model
    print("Initializing the best model (XGBoost)...")
    best_model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed, n_jobs=-1), n_jobs=-1)

    # Generate Learning Curves with extended data points up to 80k
    print("Generating saturation learning curves up to 80,000 samples. This will take a significant amount of time...")
    # Define training sizes up to 80,000
    train_sizes = np.linspace(5000, 80000, 10).astype(int)

    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, cv=3, n_jobs=-1, 
        train_sizes=train_sizes, scoring="r2"
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim Skoru")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Çapraz Doğrulama Skoru")
    
    plt.xlabel("Eğitim Örnek Sayısı")
    plt.ylabel("R² Skoru")
    plt.title('Doygunluk Noktası Analizi (80k Örneğe Kadar)')
    plt.legend(loc="best")
    plt.grid()
    
    # Save the new plot
    new_plot_path = output_dir / "learning_curves_saturation_80k.png"
    plt.savefig(new_plot_path)
    plt.close()

    print(f"\nSuccessfully generated saturation learning curve plot: {new_plot_path}")

if __name__ == '__main__':
    main()
