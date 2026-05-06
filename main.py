# main.py
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.data import random_split, DataLoader

from src.train import run_experiment
from src.dataset import ThreeChargeDataset
from src.model import ChargePredictorNet
from src.utils import (
    plot_loss,
    plot_predictions,
    plot_residuals,
    plot_prediction_distribution,
    plot_error_vs_charge_magnitude,
    save_metrics,
)
from src.extra_viz import (
    plot_parity,
    plot_residuals_v2,
    plot_residual_hist,
    plot_residual_qq,
    plot_regression_calibration,
    plot_scale_location,
    plot_residuals_vs_features,
    plot_correlation_heatmap,
    plot_scatter_matrix,
    plot_decision_tree_model,
    plot_tree_feature_importance,
    plot_tree_depth_curve,
    plot_learning_curve,
    plot_pdp_ice,
)
from src.data import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Charge Prediction Experiment")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    parser.add_argument('--with_tree', action='store_true', help='Train a DecisionTree baseline with plots')
    parser.add_argument('--extra_viz', action='store_true', help='Generate extended diagnostics')
    parser.add_argument('--metrics_txt', type=str, default=None, help='Where to save regression metrics')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=None, help='Patience for early stopping (overrides YAML)')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seed = config.get('seed', 42)
    set_seed(seed)

    dataset = ThreeChargeDataset(
        num_samples=config['samples'],
        position_range=config.get('position_range', 10.0),
        charge_range=config.get('charge_range', 5.0),
        e_min=config.get('e_min', None),
        f_min=config.get('f_min', None),
        oversample=config.get('oversample', 2),
        max_oversample=config.get('max_oversample', 64),
        seed=seed,
        min_distance=config.get('min_distance', 1e-3),
    )

    test_size = int(0.2 * len(dataset))
    val_size = int(0.1 * (len(dataset) - test_size))
    train_size = len(dataset) - test_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    batch_size = config.get('batch_size', 256)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    idx_train = np.array(train_set.indices)
    idx_val   = np.array(val_set.indices)
    idx_test  = np.array(test_set.indices)

    X_all = dataset.X
    y_all = dataset.y

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val,   y_val   = X_all[idx_val],   y_all[idx_val]
    X_test,  y_test  = X_all[idx_test],  y_all[idx_test]

    feature_names = ThreeChargeDataset.feature_names()

    model = ChargePredictorNet(
        input_dim=15,
        hidden_dims=config.get('hidden_dims', [128, 64]),
        output_dim=2,
        dropout_rate=config.get('dropout', 0.2),
        q_range=config.get('q_range', 5.0),
    ).to(device)

    if args.early_stopping:
        patience = args.patience if args.patience is not None else config.get('patience', 30)
        print(f"Early stopping ENABLED with patience={patience}")
    else:
        patience = 10**12
        print("Early stopping DISABLED (patience set very large)")

    model, history = run_experiment(
        model,
        train_loader,
        val_loader,
        epochs=config.get('epochs', 100),
        lr=config.get('lr', 1e-3),
        device=device,
        weight_decay=config.get('weight_decay', 1e-5),
        patience=patience,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_dir = Path(config.get('save_dir', './outputs'))
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y_true.append(y.numpy())
            y_pred.append(model(X).cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    errors = y_true - y_pred
    mse = np.mean(errors**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    ss_res = np.sum(errors**2, axis=0)
    r2 = 1 - ss_res / ss_tot
    print(f"Test MSE for q1: {mse[0]:.6f}, q2: {mse[1]:.6f}")
    print(f"Test R^2  for q1: {r2[0]:.6f}, q2: {r2[1]:.6f}")

    if args.metrics_txt or args.extra_viz or args.with_tree:
        metrics_path = Path(args.metrics_txt) if args.metrics_txt else (save_dir / f"metrics_{timestamp}.txt")
        save_metrics(y_true, y_pred, str(metrics_path))

    plot_loss(history, save_dir / f"loss_{timestamp}.png")
    plot_predictions(y_true, y_pred, save_dir / f"predictions_{timestamp}.png")
    plot_residuals(y_true, y_pred, save_dir / f"residuals_{timestamp}.png")
    plot_prediction_distribution(y_true, y_pred, save_dir / f"distribution_{timestamp}.png")
    plot_error_vs_charge_magnitude(y_true, y_pred, save_dir / f"error_vs_charge_{timestamp}.png")

    if args.extra_viz:
        plot_parity(y_true, y_pred, save_dir / f"parity_{timestamp}.png")
        plot_residuals_v2(y_true, y_pred, save_dir / f"residuals_vs_pred_{timestamp}.png")
        plot_residual_hist(y_true, y_pred, save_dir / f"residual_hist_{timestamp}.png")
        plot_residual_qq(y_true, y_pred, save_dir / f"residual_qq_{timestamp}.png")
        plot_regression_calibration(y_true, y_pred, save_dir / f"calibration_{timestamp}.png")
        plot_scale_location(y_true, y_pred, save_dir / f"scale_location_{timestamp}.png")
        plot_residuals_vs_features(
            X_test, y_true, y_pred, feature_names, save_dir / f"residuals_vs_features_{timestamp}.png", top_n=4
        )
        plot_correlation_heatmap(X_train, feature_names, save_dir / f"corr_heatmap_{timestamp}.png")
        plot_scatter_matrix(X_train, y_train, feature_names, save_dir / f"scatter_matrix_{timestamp}.png", top_n=4)

    if args.with_tree:
        from sklearn.tree import DecisionTreeRegressor
        tree_model = DecisionTreeRegressor(random_state=seed)
        tree_model.fit(X_train, y_train)

        plot_decision_tree_model(tree_model, feature_names, save_dir / f"tree_full_{timestamp}.png", max_depth=3)
        plot_tree_feature_importance(tree_model, feature_names, save_dir / f"tree_feature_importance_{timestamp}.png")
        plot_tree_depth_curve(
            X_train, y_train, X_val, y_val, save_dir / f"tree_depth_curve_{timestamp}.png", max_depth=10, random_state=seed
        )
        plot_learning_curve(DecisionTreeRegressor, X_train, y_train, save_dir / f"tree_learning_curve_{timestamp}.png", cv=5, random_state=seed)

        importances = getattr(tree_model, "feature_importances_", None)
        if importances is not None and importances.size > 0:
            top_indices = np.argsort(importances)[::-1][:3]
            for idx in top_indices:
                fname = feature_names[idx].replace('/', '_')
                plot_pdp_ice(tree_model, X_train, idx, save_dir / f"pdp_ice_{fname}_{timestamp}.png", feature_name=feature_names[idx])


if __name__ == '__main__':
    main()
