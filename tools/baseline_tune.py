#!/usr/bin/env python3
# baseline_tune.py — improved: JSON-safe, LINEAR/RF/GB spaces, halving optional

import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

def get_halving():
    try:
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV
        return HalvingRandomSearchCV
    except Exception:
        return None

def try_get_xgb():
    try:
        from xgboost import XGBRegressor  # type: ignore
        return XGBRegressor
    except Exception:
        return None

from src.dataset import ThreeChargeDataset
from src.data import set_seed

def json_safe(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, _np.ndarray): return o.tolist()
    return str(o)

def sanitize_params(d):
    import numpy as _np
    out = {}
    for k, v in d.items():
        if isinstance(v, (_np.integer,)): out[k] = int(v)
        elif isinstance(v, (_np.floating,)): out[k] = float(v)
        elif isinstance(v, _np.ndarray): out[k] = v.tolist()
        else: out[k] = v
    return out

def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter tuning for LINEAR/RF/GB (XGB optional)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/tuning")
    p.add_argument("--models", type=str, default="linear,rf,gb")
    p.add_argument("--n_iter", type=int, default=40)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--metric", type=str, default="r2", choices=["r2", "mse"])
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--search", type=str, default="random", choices=["random","halving"])
    return p.parse_args()

def r2_mean(y_true, y_pred):
    cols = y_true.shape[1]
    return float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(cols)]))

def neg_mse_mean(y_true, y_pred):
    cols = y_true.shape[1]
    return float(-np.mean([mean_squared_error(y_true[:, i], y_pred[:, i]) for i in range(cols)]))

SCORERS = {"r2": make_scorer(r2_mean, greater_is_better=True),
           "mse": make_scorer(neg_mse_mean, greater_is_better=True)}

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def param_distributions_for(model_name, wrapped: bool):
    prefix = "estimator__" if wrapped else ""
    if model_name == "linear":
        return {prefix + "alpha": np.logspace(-3, 3, 50)}, [prefix + "alpha"]
    if model_name == "rf":
        d = {
            prefix + "n_estimators":      np.arange(100, 1201, 100),
            prefix + "max_depth":         np.append(np.arange(3, 31, 3), [None]),
            prefix + "min_samples_split": np.arange(2, 21, 1),
            prefix + "min_samples_leaf":  np.arange(1, 21, 1),
            prefix + "max_features":      [None, "sqrt", "log2", 1.0, 0.8, 0.6, 0.4],
            prefix + "bootstrap":         [True],
        }
        return d, list(d.keys())
    if model_name == "gb":
        d = {
            prefix + "n_estimators":      np.arange(200, 1601, 100),
            prefix + "learning_rate":     np.logspace(-3, 0, 20),
            prefix + "max_depth":         np.arange(1, 8, 1),
            prefix + "subsample":         np.linspace(0.6, 1.0, 5),
            prefix + "min_samples_leaf":  np.arange(1, 21, 1),
            prefix + "n_iter_no_change":  [None, 10, 20],
            prefix + "validation_fraction":[0.1, 0.2],
        }
        return d, list(d.keys())
    if model_name == "xgb":
        d = {
            prefix + "n_estimators":      np.arange(200, 2001, 100),
            prefix + "learning_rate":     np.logspace(-3, 0, 20),
            prefix + "max_depth":         np.arange(2, 13, 1),
            prefix + "subsample":         np.linspace(0.5, 1.0, 6),
            prefix + "colsample_bytree":  np.linspace(0.5, 1.0, 6),
            prefix + "reg_lambda":        np.logspace(-3, 2, 10),
            prefix + "reg_alpha":         np.logspace(-5, 1, 12),
        }
        return d, list(d.keys())
    return {}, []

def build_estimator(model_name: str, seed=0):
    if model_name == "linear":
        return Ridge(random_state=seed)
    if model_name == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, bootstrap=True)
    if model_name == "gb":
        base = GradientBoostingRegressor(random_state=seed)
        return MultiOutputRegressor(base, n_jobs=-1)
    if model_name == "xgb":
        XGB = try_get_xgb()
        if XGB is None:
            print("[tune] xgboost not installed; skipping 'xgb'.")
            return None
        base = XGB(random_state=seed, tree_method="hist", n_jobs=-1)
        return MultiOutputRegressor(base, n_jobs=-1)
    return None

def plot_param_vs_score(df, params, score_col, outdir: Path, model_name: str, ts: str):
    for p in params:
        if p not in df.columns: continue
        vals = df[p].values; scores = df[score_col].values
        plt.figure(figsize=(6, 4))
        try:
            x = vals.astype(float)
            plt.scatter(x, scores, s=16, alpha=0.7); plt.xlabel(p)
        except Exception:
            uniq = list(dict.fromkeys([str(v) for v in vals]))
            idx = np.array([uniq.index(str(v)) for v in vals])
            plt.scatter(idx, scores, s=16, alpha=0.7)
            plt.xlabel(p + " (categorical)"); plt.xticks(range(len(uniq)), uniq, rotation=45, ha="right")
        plt.ylabel(score_col); plt.title(f"{model_name.upper()}: {p} vs {score_col}")
        plt.tight_layout(); plt.savefig(outdir / f"tune_{model_name}_{p.replace('__','_')}_{ts}.png", dpi=160); plt.close()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    X, y = ds.X, ds.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scorer = SCORERS[args.metric]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    summary = []

    use_halving = (args.search == "halving")
    Halving = get_halving()
    if use_halving and Halving is None:
        print("[tune] Halving search not available; falling back to RandomizedSearchCV.")
        use_halving = False

    for name in models:
        est = build_estimator(name, seed=seed)
        if est is None:
            continue
        wrapped = isinstance(est, MultiOutputRegressor)
        param_dist, param_keys = param_distributions_for(name, wrapped=wrapped)
        if not param_dist:
            print(f"[tune] No parameter space for '{name}'. Skipping.")
            continue

        if use_halving:
            search = Halving(
                estimator=est, param_distributions=param_dist, factor=3,
                random_state=args.random_state, scoring=scorer, cv=args.cv,
                n_jobs=-1, refit=True, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=est, param_distributions=param_dist, n_iter=args.n_iter,
                scoring=scorer, cv=args.cv, random_state=args.random_state,
                n_jobs=-1, verbose=1, refit=True, return_train_score=True,
                error_score=np.nan
            )

        search.fit(X_train, y_train)

        df = pd.DataFrame(search.cv_results_)
        df.to_csv(outdir / f"tune_results_{name}_{ts}.csv", index=False)
        plot_param_vs_score(df, param_keys, "mean_test_score", outdir, name, ts)

        best_est = search.best_estimator_
        y_pred = best_est.predict(X_test)
        mse = ((y_test - y_pred) ** 2).mean(axis=0)
        r2_each = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]

        summary.append({
            "model": name,
            "best_params": sanitize_params(search.best_params_),
            "cv_best_score": float(search.best_score_),
            "test_mse_q1": float(mse[0]), "test_mse_q2": float(mse[1]), "test_mse_mean": float(np.mean(mse)),
            "test_r2_q1": float(r2_each[0]), "test_r2_q2": float(r2_each[1]), "test_r2_mean": float(np.mean(r2_each)),
        })

        # Quick visuals
        def plot_parity(y_true, y_pred, savepath, title):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, ax in enumerate(axes):
                ax.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.6)
                lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                        max(y_true[:, i].max(), y_pred[:, i].max())]
                ax.plot(lims, lims); ax.set_xlabel(f"True q{i+1}"); ax.set_ylabel(f"Pred q{i+1}"); ax.set_title(f"Parity q{i+1}")
            plt.suptitle(title); plt.tight_layout(); plt.savefig(savepath, dpi=160); plt.close()

        def plot_residuals(y_true, y_pred, savepath, title):
            res = y_true - y_pred
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, ax in enumerate(axes):
                ax.scatter(y_pred[:, i], res[:, i], s=8, alpha=0.6)
                ax.axhline(0.0); ax.set_xlabel(f"Pred q{i+1}"); ax.set_ylabel(f"Residual q{i+1}"); ax.set_title(f"Residuals q{i+1}")
            plt.suptitle(title); plt.tight_layout(); plt.savefig(savepath, dpi=160); plt.close()

        plot_parity(y_test, y_pred, outdir / f"best_parity_{name}_{ts}.png", f"Best {name.upper()} Parity")
        plot_residuals(y_test, y_pred, outdir / f"best_residuals_{name}_{ts}.png", f"Best {name.upper()} Residuals")

    with open(outdir / f"tune_summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_safe, ensure_ascii=False)

    print("=== Hyperparameter Tuning Summary ===")
    for row in summary:
        print(f"{row['model'].upper()} | CV best={row['cv_best_score']:.4f} | Test R2(mean)={row['test_r2_mean']:.4f} | Test MSE(mean)={row['test_mse_mean']:.4e}")
        print(f"  Best params: {row['best_params']}")
    print(f"Saved artifacts to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
