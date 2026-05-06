"""
extra_viz.py
-----------------

This module contains a collection of diagnostic plots and helper functions for
regression models. The functions defined here are intended to supplement the
baseline visualisations used in ``utils.py`` by providing deeper insights
into model performance, error structure, feature relationships and model
complexity. Nothing in this file modifies or depends on internal state of
other modules – everything is self‑contained. You can import individual
functions as needed in your training or evaluation scripts.

Overview of available plots:

* **Parity plot** – scatter plot of true targets vs predictions with
  identity line. Useful for spotting bias and scale issues.
* **Residuals vs predicted** – scatter of fitted values vs residuals; a
  horizontal band indicates homoskedasticity.
* **Residual histogram** – distribution of residuals for each output.
* **Q–Q plot** – quantile–quantile plot comparing residuals against a
  theoretical normal distribution.
* **Regression calibration** – binned reliability diagram for continuous
  targets. Plots mean prediction vs mean observed value in bins of
  predicted values.
* **Scale–location plot** – square root of absolute standardised residuals
  vs fitted values. Highlights heteroskedasticity.
* **Residuals vs features** – scatter of selected features vs residuals,
  showing whether errors correlate with particular inputs.
* **Correlation heatmap** – heatmap of Pearson correlations between
  features.
* **Scatter matrix (pair plot)** – pairwise scatter plots of selected
  influential features.
* **Decision tree visualisations** – functions to visualise a
  ``DecisionTreeRegressor``: full tree diagram, feature importances,
  depth vs R² curves, learning curves and PDP/ICE plots.

The functions here depend only on widely available packages such as
NumPy, SciPy, scikit‑learn, pandas, seaborn and matplotlib. Where a
dependency is optional (e.g. SciPy for Q–Q plots), the code attempts
to fail gracefully if not installed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, List, Optional

# Try to import seaborn if available; fall back to matplotlib
try:
    import seaborn as sns  # type: ignore
except ImportError:
    sns = None

from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.inspection import partial_dependence
from sklearn.exceptions import NotFittedError

# Import helper from utils to ensure output directories exist
from src.utils import ensure_dir


def _standardised_residuals(residuals: np.ndarray) -> np.ndarray:
    """Return standardised residuals for each output column.

    Parameters
    ----------
    residuals : ndarray of shape (n_samples, n_outputs)
        Raw residuals (y_true - y_pred).

    Returns
    -------
    ndarray
        Standardised residuals for each output.
    """
    # Standard deviation of residuals for each output
    std = np.std(residuals, axis=0, ddof=1)
    # Avoid divide by zero
    std = np.where(std == 0, 1.0, std)
    return residuals / std


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    """Plot parity (true vs predicted) for each output dimension.

    A perfect model would lie on the diagonal y=x. The scatter helps to
    visualise bias (vertical offset) and variance (spread). A dashed red
    identity line is drawn for reference.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_outputs)
        Ground truth values.
    y_pred : ndarray of shape (n_samples, n_outputs)
        Predicted values.
    save_path : str or Path
        Destination file path for the plot.
    """
    ensure_dir(save_path)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    for i in range(n_outputs):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.4)
        # diagonal line
        min_val = np.min(np.concatenate([y_true[:, i], y_pred[:, i]]))
        max_val = np.max(np.concatenate([y_true[:, i], y_pred[:, i]]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax.set_xlabel(f"True value (output {i})")
        ax.set_ylabel(f"Predicted value (output {i})")
        ax.set_title(f"Parity plot for output {i}")
        ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_residuals_v2(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    """Plot residuals versus predicted values for each output.

    This scatter plot helps detect non‑constant variance and non‑linear
    relationships between the predictions and the errors. A horizontal
    reference line at zero residual is drawn.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_outputs)
        Ground truth values.
    y_pred : ndarray of shape (n_samples, n_outputs)
        Predicted values.
    save_path : str or Path
        Destination file path for the plot.
    """
    ensure_dir(save_path)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    for i in range(n_outputs):
        ax = axes[i]
        ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.4)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel(f"Predicted value (output {i})")
        ax.set_ylabel(f"Residual (output {i})")
        ax.set_title(f"Residuals vs Predicted for output {i}")
        ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path, bins: int = 50) -> None:
    """Plot histogram of residuals for each output.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_outputs)
        Ground truth values.
    y_pred : ndarray of shape (n_samples, n_outputs)
        Predicted values.
    save_path : str or Path
        Destination file path for the plot.
    bins : int, optional
        Number of histogram bins (default=50).
    """
    ensure_dir(save_path)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    n_outputs = residuals.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    for i in range(n_outputs):
        ax = axes[i]
        ax.hist(residuals[:, i], bins=bins, alpha=0.7, color='grey', edgecolor='black')
        ax.set_title(f"Residual histogram for output {i}")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_residual_qq(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    """Create Q–Q plots comparing residuals to a standard normal distribution.

    If SciPy is available, ``scipy.stats.probplot`` is used to compute the
    theoretical and sample quantiles. Otherwise the function attempts to
    approximate the quantiles manually. Each output dimension is plotted in its
    own subplot.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted target values.
    save_path : str or Path
        Destination file path for the plot.
    """
    ensure_dir(save_path)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    n_outputs = residuals.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    try:
        from scipy import stats  # type: ignore
        use_scipy = True
    except Exception:
        use_scipy = False
    for i in range(n_outputs):
        ax = axes[i]
        res = residuals[:, i]
        res = res[np.isfinite(res)]
        if use_scipy:
            (theo_q, samp_q), (slope, intercept, r) = stats.probplot(res, dist="norm")
        else:
            # Manual quantile calculation: sort residuals and match with
            # theoretical quantiles of standard normal
            n = len(res)
            samp_q = np.sort(res)
            # rank based probabilities (i-0.5)/n
            p = (np.arange(1, n + 1) - 0.5) / n
            from math import sqrt
            from numpy import log as ln
            # approximate inverse CDF of normal via erfinv or stats.norm.ppf if not available
            try:
                from scipy.special import erfinv  # type: ignore
                theo_q = sqrt(2) * erfinv(2 * p - 1)
            except Exception:
                # Fallback: use approximation of inverse CDF via a rational approximation
                # of the error function; accuracy is adequate for diagnostic plot
                # Source: Winitzki, S. (2008). A handy approximation for the error function.
                a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
                x = 2 * p - 1
                sign = np.sign(x)
                ln_term = np.log(1 - x**2)
                theo_q = sign * np.sqrt(
                    np.sqrt( (2 / (np.pi * a) + ln_term / 2)**2 - ln_term / a ) - (2 / (np.pi * a) + ln_term / 2)
                )
            slope = 1.0
            intercept = 0.0
        ax.scatter(theo_q, samp_q, alpha=0.5, edgecolor='k', facecolor='none')
        # reference line
        min_q = min(theo_q.min(), samp_q.min())
        max_q = max(theo_q.max(), samp_q.max())
        ax.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=1)
        ax.set_title(f"Q–Q plot for residuals (output {i})")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_regression_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Plot a binned calibration diagram for continuous regression.

    The predicted values are binned into quantiles; for each bin, the mean
    predicted value and mean true value are computed. A perfectly calibrated
    model would lie on the y=x line. Inspired by classification reliability
    diagrams but adapted for regression.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, n_outputs)
        Ground truth target values.
    y_pred : ndarray, shape (n_samples, n_outputs)
        Predicted target values.
    save_path : str or Path
        File path to save the plot.
    n_bins : int, optional
        Number of bins (default=10).
    """
    ensure_dir(save_path)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    for i in range(n_outputs):
        ax = axes[i]
        preds = y_pred[:, i]
        trues = y_true[:, i]
        # Compute bin edges using quantiles to ensure equal counts
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(preds, quantiles)
        # To avoid duplicate edges due to many identical predictions
        bins = np.unique(bins)
        # Assign each prediction to a bin index
        inds = np.digitize(preds, bins, right=True) - 1
        # Clip indices that fall outside
        inds = np.clip(inds, 0, len(bins) - 2)
        mean_pred = []
        mean_true = []
        for b in range(len(bins) - 1):
            mask = inds == b
            if np.any(mask):
                mean_pred.append(np.mean(preds[mask]))
                mean_true.append(np.mean(trues[mask]))
        ax.plot(mean_pred, mean_true, marker='o', linestyle='-', label='Binned means')
        # Identity line
        all_vals = np.concatenate([y_true[:, i], y_pred[:, i]])
        min_val = np.min(all_vals)
        max_val = np.max(all_vals)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Ideal')
        ax.set_xlabel(f"Mean predicted (output {i})")
        ax.set_ylabel(f"Mean true (output {i})")
        ax.set_title(f"Regression calibration for output {i}")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# src/extra_viz.py
def plot_scale_location(y_true, y_pred, out_path, use_mathtext: bool = False):
    """
    Scale–Location (spread–level) plot:
      x = predicted (fitted) values
      y = sqrt(|standardized residuals|)
    Not: MathText'te çıplak '|' sorun çıkarır; düz metin veya \lvert...\rvert kullan.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    resid = y_true - y_pred
    mu = float(np.mean(resid))
    sd = float(np.std(resid, ddof=1))
    # aşırı küçük/NaN std için güvenlik
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1e-12

    z = (resid - mu) / sd
    y_scatter = np.sqrt(np.abs(z))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, y_scatter, s=10, alpha=0.5)
    ax.set_xlabel("Predicted")

    if use_mathtext:
        # Güvenli mathtext sürümü (çıplak | yerine \lvert ... \rvert)
        ax.set_ylabel(r"$\sqrt{\lvert \mathrm{standardized\ residuals} \rvert}$")
    else:
        # En güvenlisi: düz metin
        ax.set_ylabel("sqrt(|standardized residuals|)")

    ax.set_title("Scale–Location (spread–level) plot")
    ax.grid(True, alpha=0.2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)



def plot_residuals_vs_features(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    save_path: str | Path,
    top_n: int = 4,
) -> None:
    """Plot residuals against the most influential features.

    The function computes the absolute Pearson correlation between each feature
    and the residuals (across all outputs) and selects the top ``top_n``
    features. It then produces a grid of scatter plots: each column
    corresponds to a selected feature and each row corresponds to an output
    dimension. The correlation value is shown in the subplot title to
    indicate the strength of association.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input features (scaled or raw).
    y_true : ndarray, shape (n_samples, n_outputs)
        Ground truth target values.
    y_pred : ndarray, shape (n_samples, n_outputs)
        Predicted target values.
    feature_names : list of str
        Names of each feature; length must equal number of columns in ``X``.
    save_path : str or Path
        Output file path for the plot.
    top_n : int, optional
        Number of top correlated features to display (default=4).
    """
    ensure_dir(save_path)
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    n_features = X.shape[1]
    n_outputs = residuals.shape[1]
    # Compute correlation between each feature and residuals across outputs; take max absolute
    corr_vals = []
    for i in range(n_features):
        # handle constant features
        xi = X[:, i]
        max_corr = 0.0
        for j in range(n_outputs):
            resj = residuals[:, j]
            if np.std(xi) == 0 or np.std(resj) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(xi, resj)[0, 1]
            max_corr = max(max_corr, abs(corr))
        corr_vals.append(max_corr)
    # select top features
    idx_sorted = np.argsort(corr_vals)[::-1]
    top_idx = idx_sorted[:min(top_n, len(idx_sorted))]
    # Set up subplots: n_outputs rows x top_n columns
    fig, axes = plt.subplots(n_outputs, len(top_idx), figsize=(5 * len(top_idx), 4 * n_outputs), squeeze=False)
    for row in range(n_outputs):
        for col, feat_idx in enumerate(top_idx):
            ax = axes[row, col]
            xi = X[:, feat_idx]
            resj = residuals[:, row]
            ax.scatter(xi, resj, alpha=0.4)
            # compute correlation value for this output
            if np.std(xi) == 0 or np.std(resj) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(xi, resj)[0, 1]
            ax.set_title(f"{feature_names[feat_idx]}\nρ={corr:.2f}")
            ax.set_xlabel(feature_names[feat_idx])
            ax.set_ylabel(f"Residual (out {row})")
            ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_correlation_heatmap(
    X: np.ndarray,
    feature_names: List[str],
    save_path: str | Path,
    annot: bool = False,
) -> None:
    """Plot a Pearson correlation heatmap of the input features.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix (scaled or raw).
    feature_names : list of str
        Names of each feature.
    save_path : str or Path
        File path for saving the heatmap.
    annot : bool, optional
        Whether to annotate each cell with the correlation value (default=False).
    """
    ensure_dir(save_path)
    X = np.asarray(X)
    n_features = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(0.6 * n_features + 3, 0.6 * n_features + 3))
    if sns is not None:
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, xticklabels=feature_names, yticklabels=feature_names,
                    ax=ax, cmap=cmap, center=0, annot=annot, fmt=".2f" if annot else None)
    else:
        # fallback to matplotlib's imshow
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(feature_names)
        fig.colorbar(im, ax=ax)
        if annot:
            for i in range(n_features):
                for j in range(n_features):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color='k', fontsize=8)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_scatter_matrix(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_path: str | Path,
    top_n: int = 4,
) -> None:
    """Create a scatter matrix (pair plot) for the most correlated features.

    The function selects ``top_n`` features that exhibit the strongest
    correlation (by absolute value) with the target outputs. It then
    constructs a pandas DataFrame and leverages seaborn’s ``pairplot``
    (if available) or ``pandas.plotting.scatter_matrix`` as a fallback.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input feature matrix.
    y : ndarray, shape (n_samples, n_outputs)
        Target values used to compute feature relevance.
    feature_names : list of str
        Names of each feature.
    save_path : str or Path
        Destination path for the generated plot.
    top_n : int, optional
        Number of features to include (default=4).
    """
    ensure_dir(save_path)
    X = np.asarray(X)
    y = np.asarray(y)
    n_features = X.shape[1]
    n_outputs = y.shape[1]
    # Compute relevance of each feature as max absolute correlation with any output
    rels = []
    for i in range(n_features):
        xi = X[:, i]
        max_corr = 0.0
        for j in range(n_outputs):
            yj = y[:, j]
            if np.std(xi) == 0 or np.std(yj) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(xi, yj)[0, 1]
            max_corr = max(max_corr, abs(corr))
        rels.append(max_corr)
    idx_sorted = np.argsort(rels)[::-1]
    top_idx = idx_sorted[:min(top_n, len(idx_sorted))]
    import pandas as pd
    cols = {feature_names[i]: X[:, i] for i in top_idx}
    df = pd.DataFrame(cols)
    # If seaborn is available, use pairplot for nicer aesthetics
    fig = None
    if sns is not None:
        pairplot = sns.pairplot(df, diag_kind='hist')
        # Save pairplot figure
        pairplot.fig.suptitle("Scatter matrix for top features", y=1.02)
        pairplot.fig.savefig(save_path)
        plt.close(pairplot.fig)
    else:
        # Fallback: use pandas' scatter_matrix
        from pandas.plotting import scatter_matrix
        fig = plt.figure()
        scatter_matrix(df, alpha=0.3, figsize=(4 * len(top_idx), 4 * len(top_idx)), diagonal='hist')
        plt.suptitle("Scatter matrix for top features")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


def plot_decision_tree_model(
    tree: DecisionTreeRegressor,
    feature_names: List[str],
    save_path: str | Path,
    max_depth: Optional[int] = None,
) -> None:
    """Visualise a fitted ``DecisionTreeRegressor``.

    A simplified tree diagram is drawn using scikit‑learn’s ``plot_tree``.
    If ``max_depth`` is specified, only the top levels up to that depth are
    displayed to improve readability. Feature names and splitting thresholds
    are annotated automatically by scikit‑learn.

    Parameters
    ----------
    tree : DecisionTreeRegressor
        A fitted decision tree.
    feature_names : list of str
        Names of each feature.
    save_path : str or Path
        File path for saving the diagram.
    max_depth : int, optional
        Maximum depth of the plotted tree. If ``None``, the full tree is
        rendered (may be very large).
    """
    ensure_dir(save_path)
    if not hasattr(tree, "tree_"):
        raise NotFittedError("DecisionTreeRegressor instance is not fitted yet")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(tree, feature_names=feature_names, filled=True, impurity=False,
              max_depth=max_depth, ax=ax)
    ax.set_title("Decision tree diagram")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_tree_feature_importance(
    tree: DecisionTreeRegressor,
    feature_names: List[str],
    save_path: str | Path
) -> None:
    """Plot the feature importances of a decision tree.

    Parameters
    ----------
    tree : DecisionTreeRegressor
        A fitted decision tree.
    feature_names : list of str
        Names of each feature.
    save_path : str or Path
        File path to save the bar chart.
    """
    ensure_dir(save_path)
    importances = getattr(tree, "feature_importances_", None)
    if importances is None:
        raise NotFittedError("DecisionTreeRegressor instance is not fitted yet")
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_names)
    ax.set_ylabel("Importance")
    ax.set_title("Feature importances (Decision Tree)")
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_tree_depth_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: str | Path,
    max_depth: int = 10,
    random_state: Optional[int] = None,
) -> None:
    """Plot training and validation R² scores versus tree depth.

    The function fits decision trees of increasing depth on the training data
    and evaluates R² on both the training and validation sets. Plotting the
    curves helps diagnose overfitting (large gaps between train and val) and
    underfitting (both scores low).

    Parameters
    ----------
    X_train, y_train : ndarray
        Training data.
    X_val, y_val : ndarray
        Validation data.
    save_path : str or Path
        Where to save the plot.
    max_depth : int, optional
        Maximum depth of trees to evaluate (default=10).
    random_state : int, optional
        Random state for reproducibility.
    """
    ensure_dir(save_path)
    depths = list(range(1, max_depth + 1))
    train_scores: List[float] = []
    val_scores: List[float] = []
    for d in depths:
        tree = DecisionTreeRegressor(max_depth=d, random_state=random_state)
        tree.fit(X_train, y_train)
        # multioutput R²
        y_pred_train = tree.predict(X_train)
        y_pred_val = tree.predict(X_val)
        train_scores.append(r2_score(y_train, y_pred_train))
        val_scores.append(r2_score(y_val, y_pred_val))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(depths, train_scores, marker='o', label='Train R²')
    ax.plot(depths, val_scores, marker='s', label='Validation R²')
    ax.set_xlabel("Max depth")
    ax.set_ylabel("R² score")
    ax.set_title("Decision tree depth vs R²")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_learning_curve(
    estimator_cls,
    X: np.ndarray,
    y: np.ndarray,
    save_path: str | Path,
    cv: int = 5,
    train_sizes: Optional[Iterable[float]] = None,
    random_state: Optional[int] = None,
) -> None:
    """Generate a learning curve for a given estimator class.

    This function uses scikit‑learn’s ``learning_curve`` utility to compute
    training and cross‑validation R² scores for different fractions of the
    training set. It assumes that the estimator supports multioutput and
    accepts a ``random_state`` argument in its constructor (if provided).

    Parameters
    ----------
    estimator_cls : class
        A scikit‑learn regressor class (not an instance). Should support
        ``random_state`` in constructor for reproducibility.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    y : ndarray, shape (n_samples, n_outputs)
        Target data.
    save_path : str or Path
        Destination for the plot.
    cv : int, optional
        Number of cross‑validation folds (default=5).
    train_sizes : iterable of float, optional
        Relative sizes of the training set to use (default is 5 values from
        10% to 100%). Values should be between 0 and 1.
    random_state : int, optional
        Random state for reproducibility. If provided, passed to the
        estimator’s constructor.
    """
    ensure_dir(save_path)
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    # Build estimator with optional random_state
    def make_est():
        try:
            return estimator_cls(random_state=random_state)
        except TypeError:
            return estimator_cls()
    # Use scikit‑learn's learning_curve; scoring uses R² for regression
    est = make_est()
    train_sizes_abs, train_scores, val_scores = learning_curve(
        est, X, y, train_sizes=train_sizes, scoring='r2', cv=cv, shuffle=True, random_state=random_state
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes_abs, train_mean, 'o-', label='Train R²')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes_abs, val_mean, 's-', label='CV R²')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("R² score")
    ax.set_title(f"Learning curve ({estimator_cls.__name__})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pdp_ice(
    model,
    X: np.ndarray,
    feature_index: int,
    save_path: str | Path,
    feature_name: Optional[str] = None,
    n_points: int = 25,
    n_samples: int = 50,
) -> None:
    """Plot partial dependence and individual conditional expectation (ICE).

    For the specified feature, the function draws two subplots, one for
    each output. The thick line represents the averaged partial dependence,
    whereas thin lines depict ICE curves for a subset of samples. When
    scikit‑learn’s ``partial_dependence`` is available, it is used for the
    average; otherwise the average is computed manually.

    Parameters
    ----------
    model : estimator
        A fitted regressor supporting multioutput predictions via ``predict``.
    X : ndarray, shape (n_samples_total, n_features)
        Data on which to compute partial dependence.
    feature_index : int
        Index of the feature for which to compute the dependence.
    save_path : str or Path
        Destination for the figure.
    feature_name : str, optional
        Name of the feature (for labelling); if ``None``, uses index.
    n_points : int, optional
        Number of points in the feature grid (default=25).
    n_samples : int, optional
        Number of samples to plot ICE curves (default=50).
    """
    ensure_dir(save_path)
    X = np.asarray(X)
    feature_name = feature_name or f"feature {feature_index}"
    # Determine grid for the feature
    f_vals = X[:, feature_index]
    f_min, f_max = np.min(f_vals), np.max(f_vals)
    grid = np.linspace(f_min, f_max, n_points)
    # Sample rows for ICE curves
    total_samples = X.shape[0]
    if n_samples > total_samples:
        sample_indices = np.arange(total_samples)
    else:
        rng = np.random.default_rng(0)
        sample_indices = rng.choice(total_samples, size=n_samples, replace=False)
    # Precompute baseline values for other features
    X_baseline = X.copy()
    # Determine number of outputs
    try:
        # If model exposes n_outputs_ attribute
        n_outputs = model.n_outputs_
    except Exception:
        # Fallback: use prediction shape
        n_outputs = model.predict(X[:1]).shape[1]
    # Prepare figure: two rows if there are two outputs
    fig, axes = plt.subplots(n_outputs, 1, figsize=(7, 4 * n_outputs), squeeze=False)
    for output_idx in range(n_outputs):
        ax = axes[output_idx, 0]
        # Compute average (PDP)
        try:
            pdp_result = partial_dependence(model, X, [feature_index], grid_resolution=n_points)
            # pdp_result averaged results at axis 0 (for n_outputs)
            avg = pdp_result.average[output_idx].ravel()
        except Exception:
            # Manual computation: vary feature across grid, average predictions
            avg_vals = []
            for val in grid:
                X_temp = X.copy()
                X_temp[:, feature_index] = val
                preds = model.predict(X_temp)
                avg_vals.append(np.mean(preds[:, output_idx]))
            avg = np.array(avg_vals)
        ax.plot(grid, avg, color='black', linewidth=2, label='Partial dependence')
        # Plot ICE curves
        for idx in sample_indices:
            x0 = X_baseline[idx].copy()
            ice_vals = []
            for val in grid:
                x_temp = x0.copy()
                x_temp[feature_index] = val
                pred = model.predict(x_temp.reshape(1, -1))[0, output_idx]
                ice_vals.append(pred)
            ax.plot(grid, ice_vals, color='grey', alpha=0.2)
        ax.set_xlabel(feature_name)
        ax.set_ylabel(f"Predicted output {output_idx}")
        ax.set_title(f"PDP/ICE for {feature_name} (output {output_idx})")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)