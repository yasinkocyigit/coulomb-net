
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
    parser = argparse.ArgumentParser(description="Generate Extended Learning Curves")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    config = load_config(args.config)

    # Setup directories and seed
    output_dir = Path("C:/Users/yasin/Documents/projectv10/projectv10/output_plt2")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.get('seed', 42)
    set_seed(seed)

    # Load a larger dataset to accommodate up to 40k training samples
    # We need 50k total samples to get 40k for training (80% split)
    print("Generating a larger dataset (50,000 samples)...")
    dataset = ThreeChargeDataset(
        num_samples=60000, # Increased sample size to 60k
        position_range=config.get('position_range', 10.0),
        charge_range=config.get('charge_range', 5.0),
        seed=seed
    )
    X, y = dataset.X, dataset.y

    # Define the best model
    print("Initializing the best model (XGBoost)...")
    best_model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed, n_jobs=-1), n_jobs=-1)

    # Generate Learning Curves with extended data points
    print("Generating extended learning curves up to 40,000 samples...")
    # Define training sizes up to 40,000
    train_sizes = np.linspace(2500, 40000, 8).astype(int)

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
    plt.title('Genişletilmiş Öğrenme Eğrileri (40k Örneğe Kadar)')
    plt.legend(loc="best")
    plt.grid()
    
    # Save the new plot
    new_plot_path = output_dir / "learning_curves_extended_40k.png"
    plt.savefig(new_plot_path)
    plt.close()

    print(f"\nSuccessfully generated new learning curve plot: {new_plot_path}")

if __name__ == '__main__':
    main()
