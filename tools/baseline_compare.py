#!/usr/bin/env python3
# baseline_compare.py — improved: RidgeCV for linear, RF/GB sane defaults, early stopping for GB,
# fast mode, learning-curve controls, and feature-importance/coefficient plots.

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

def try_get_xgb():
    try:
        from xgboost import XGBRegressor  # type: ignore
        return XGBRegressor
    except Exception:
        return None

from src.dataset import ThreeChargeDataset
from src.data import set_seed

def parse_args():
    p = argparse.ArgumentParser(description="Sklearn baselines (Linear/RF/GB/XGB optional) for charge prediction")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/baselines")
    p.add_argument("--models", type=str, default="linear,rf,gb")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--fast", action="store_true", help="Skip/trim heavy plots to speed up")
    return p.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_models(names, seed=0):
    names = [n.strip().lower() for n in names.split(",") if n.strip()]
    models = {}
    for n in names:
        if n == "linear":
            models["linear"] = RidgeCV(alphas=np.logspace(-3, 3, 13), fit_intercept=True)
        elif n == "rf":
            models["rf"] = RandomForestRegressor(
                n_estimators=300, max_depth=None, max_features="sqrt",
                bootstrap=True, n_jobs=-1, random_state=seed
            )
        elif n == "gb":
            base = GradientBoostingRegressor(
                n_estimators=800, learning_rate=0.05, max_depth=3,
                subsample=0.8, min_samples_leaf=5,
                n_iter_no_change=20, validation_fraction=0.1,
                random_state=seed
            )
            models["gb"] = MultiOutputRegressor(base, n_jobs=-1)
        elif n == "xgb":
            XGB = try_get_xgb()
            if XGB is not None:
                base = XGB(n_estimators=800, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, reg_alpha=0.0,
                           random_state=seed, tree_method="hist", n_jobs=-1)
                models["xgb"] = MultiOutputRegressor(base, n_jobs=-1)
            else:
                print("[baseline_compare] xgboost not installed; skipping 'xgb'.")
        else:
            print(f"[baseline_compare] Unknown model name: {n} (skipping)")
    return models

def plot_parity(y_true, y_pred, savepath, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.6)
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        ax.plot(lims, lims)
        ax.set_xlabel(f"True q{i+1}"); ax.set_ylabel(f"Pred q{i+1}"); ax.set_title(f"Parity q{i+1}")
    plt.suptitle(title); plt.tight_layout(); plt.savefig(savepath, dpi=160); plt.close()

def plot_residuals(y_true, y_pred, savepath, title):
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, ax in enumerate(axes):
        ax.scatter(y_pred[:, i], res[:, i], s=8, alpha=0.6)
        ax.axhline(0.0); ax.set_xlabel(f"Pred q{i+1}"); ax.set_ylabel(f"Residual q{i+1}"); ax.set_title(f"Residuals q{i+1}")
    plt.suptitle(title); plt.tight_layout(); plt.savefig(savepath, dpi=160); plt.close()

def plot_learning_curve_wrapper(estimator, X, y, savepath, title, cv=5, fast=False):
    train_sizes = np.linspace(0.5, 1.0, 3) if fast else np.linspace(0.1, 1.0, 6)
    tr_sizes, tr_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="r2", n_jobs=-1, train_sizes=train_sizes, shuffle=True, random_state=0
    )
    tr_mean = tr_scores.mean(axis=1); tr_std = tr_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1); val_std = val_scores.std(axis=1)
    plt.figure(figsize=(6, 4))
    plt.fill_between(tr_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)
    plt.fill_between(tr_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.plot(tr_sizes, tr_mean, marker="o", label="Train R^2")
    plt.plot(tr_sizes, val_mean, marker="s", label="CV R^2")
    plt.xlabel("Training examples"); plt.ylabel("R^2 score")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(savepath, dpi=160); plt.close()

def plot_feature_importance(model_name, estimator, feature_names, savepath):
    fig = plt.figure(figsize=(7, 4))
    if model_name == "rf":
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None: return
        idx = np.argsort(importances)[::-1][:10]
        labels = [feature_names[i] for i in idx]
        plt.barh(range(len(idx)), importances[idx][::-1])
        plt.yticks(range(len(idx)), labels[::-1])
        plt.title("RF: Top-10 Feature Importances")
    elif model_name == "gb":
        ests = getattr(estimator, "estimators_", None)
        if not ests: return
        imps = []
        for base in ests:
            imp = getattr(base, "feature_importances_", None)
            if imp is not None: imps.append(imp)
        if not imps: return
        importances = np.mean(np.vstack(imps), axis=0)
        idx = np.argsort(importances)[::-1][:10]
        labels = [feature_names[i] for i in idx]
        plt.barh(range(len(idx)), importances[idx][::-1])
        plt.yticks(range(len(idx)), labels[::-1])
        plt.title("GB: Top-10 Feature Importances (avg over targets)")
    elif model_name == "linear":
        coefs = getattr(estimator, "coef_", None)
        if coefs is None: return
        for t in range(coefs.shape[0]):
            plt.clf()
            coef = coefs[t]
            idx = np.argsort(np.abs(coef))[::-1][:10]
            labels = [feature_names[i] for i in idx]
            vals = coef[idx]
            plt.barh(range(len(idx)), vals[::-1])
            plt.yticks(range(len(idx)), labels[::-1])
            plt.title(f"Ridge: Top-10 Coefficients (target q{t+1})")
            plt.tight_layout()
            plt.savefig(savepath.parent / f"{savepath.stem}_q{t+1}{savepath.suffix}", dpi=160)
        return
    else:
        return
    plt.tight_layout()
    fig.savefig(savepath, dpi=160)
    plt.close(fig)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = cfg.get("seed", 42); set_seed(seed)

    ds = ThreeChargeDataset(
        num_samples=cfg["samples"],
        position_range=cfg.get("position_range", 10.0),
        charge_range=cfg.get("charge_range", 5.0),
        e_min=cfg.get("e_min", None),
        f_min=cfg.get("f_min", None),
        oversample=cfg.get("oversample", 2),
        max_oversample=cfg.get("max_oversample", 64),
        seed=seed,
        min_distance=cfg.get("min_distance", 1e-3),
    )

    X_all, y_all = ds.X, ds.y
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=seed)

    feature_names = ThreeChargeDataset.feature_names()
    models = get_models(args.models, seed=seed)
    results = []

    for name, model in models.items():
        est = model
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)

        mse = ((y_test - y_pred) ** 2).mean(axis=0)
        mae = np.mean(np.abs(y_test - y_pred), axis=0)
        medae = np.array([median_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
        r2_each = np.array([r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])

        results.append({
            "model": name,
            "mse_q1": float(mse[0]), "mse_q2": float(mse[1]), "mse_mean": float(mse.mean()),
            "mae_q1": float(mae[0]), "mae_q2": float(mae[1]), "mae_mean": float(mae.mean()),
            "medae_q1": float(medae[0]), "medae_q2": float(medae[1]), "medae_mean": float(medae.mean()),
            "r2_q1": float(r2_each[0]), "r2_q2": float(r2_each[1]), "r2_mean": float(r2_each.mean()),
        })

        plot_parity(y_test, y_pred, outdir / f"parity_{name}_{timestamp}.png", f"Parity: {name.upper()}")
        plot_residuals(y_test, y_pred, outdir / f"residuals_{name}_{timestamp}.png", f"Residuals: {name.upper()}")
        if not args.fast:
            plot_learning_curve_wrapper(est, X_train, y_train, outdir / f"learning_curve_{name}_{timestamp}.png",
                                        f"Learning Curve: {name.upper()}", cv=args.cv, fast=args.fast)

        plot_feature_importance(name, est, feature_names, outdir / f"importance_{name}_{timestamp}.png")

    import csv, json
    if results:
        csv_path = outdir / f"baseline_results_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader(); writer.writerows(results)
        with open(outdir / f"baseline_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

    if results:
        print("=== Baseline Results ===")
        for row in results:
            print(f"{row['model']:>6s} | R2(mean)={row['r2_mean']:.4f} | MSE(mean)={row['mse_mean']:.4e} | MAE(mean)={row['mae_mean']:.4e} | MedAE(mean)={row['medae_mean']:.4e}")
        print(f"Saved artifacts to: {outdir.resolve()}")
    else:
        print("No models were run. (Did you request 'xgb' without installing xgboost?)")

if __name__ == "__main__":
    main()
