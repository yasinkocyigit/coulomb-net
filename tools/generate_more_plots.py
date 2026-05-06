import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.dataset import ThreeChargeDataset
from src.utils import ensure_dir
import statsmodels.api as sm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def plot_hexbin_density(x, y, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 6))
    sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")
    plt.savefig(save_path)
    plt.close()

def plot_ecdf_with_ci(data, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 6))
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(min(data), max(data))
    y = ecdf(x)
    plt.step(x, y, where='post')
    # This is a simplified CI. For a more accurate one, bootstrapping would be needed.
    n = len(data)
    ci_epsilon = 1.36 / np.sqrt(n)
    plt.fill_between(x, np.maximum(0, y - ci_epsilon), np.minimum(1, y + ci_epsilon), alpha=0.3)
    plt.xlabel("Value")
    plt.ylabel("ECDF")
    plt.title("ECDF with 95% CI")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_violin_with_medians(data, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, inner="quartile")
    plt.title("Violin Plot with Medians")
    plt.savefig(save_path)
    plt.close()

def plot_waterfall(feature_names, contributions, save_path):
    ensure_dir(save_path)
    
    # Sort features by contribution
    sorted_indices = np.argsort(np.abs(contributions))[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_contributions = contributions[sorted_indices]

    cumulative = np.cumsum(sorted_contributions)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(sorted_names, sorted_contributions)
    ax.plot(sorted_names, cumulative, marker='o', color='r', label='Cumulative')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Contribution")
    plt.title("Feature Importance Waterfall Chart")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_contour_map(x, y, z, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(x, y, z, levels=14, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Contour Map")
    plt.savefig(save_path)
    plt.close()

def plot_pareto_frontier(costs, benefits, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 6))
    # Sort by costs
    sorted_indices = np.argsort(costs)
    costs = costs[sorted_indices]
    benefits = benefits[sorted_indices]
    
    # Identify pareto points
    pareto_front = [True] * len(costs)
    for i in range(len(costs)):
        for j in range(len(costs)):
            if i != j and costs[j] <= costs[i] and benefits[j] >= benefits[i]:
                pareto_front[i] = False
                break
    
    plt.scatter(costs, benefits, c=pareto_front, cmap="viridis")
    plt.xlabel("Costs")
    plt.ylabel("Benefits")
    plt.title("Pareto Frontier")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_scatter_with_covariance_ellipse(x, y, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x, y=y, scatter_kws={'alpha':0.3})
    # Covariance ellipse
    from matplotlib.patches import Ellipse
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ax = plt.gca()
    for j in range(1, 4):
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      edgecolor='red', facecolor='none')
        ax.add_artist(ell)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter with Covariance Ellipse")
    plt.savefig(save_path)
    plt.close()


def plot_fan_chart(x, y_preds, save_path):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 6))
    
    mean_preds = np.mean(y_preds, axis=0)
    percentiles = np.percentile(y_preds, [5, 25, 75, 95], axis=0)
    
    plt.plot(x, mean_preds, label='Mean Prediction')
    plt.fill_between(x, percentiles[0], percentiles[3], alpha=0.2, label='90% CI')
    plt.fill_between(x, percentiles[1], percentiles[2], alpha=0.4, label='50% CI')
    
    plt.xlabel("Feature Value")
    plt.ylabel("Prediction")
    plt.title("Fan Chart")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_phase_map(model, X, y, save_path):
    ensure_dir(save_path)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Threshold the target variable to create a binary classification problem
    y_binary = (y > y.mean()).astype(int)
    model.fit(X, y_binary)
    
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_binary, s=20, edgecolor='k')
    plt.title("Phase Map with Decision Boundary")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(save_path)
    plt.close()

def plot_radar_chart(metrics, model_names, save_path):
    ensure_dir(save_path)
    from math import pi

    labels = list(metrics.keys())
    num_vars = len(labels)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, model_name in enumerate(model_names):
        values = [metrics[label][i] for label in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.title("Model Comparison Radar Chart", size=11, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(save_path)
    plt.close()


def plot_vector_field(save_path):
    ensure_dir(save_path)
    # Define a simple 2D function
    def f(x, y):
        return x**2 + y**2

    # Define the gradient of the function
    def grad_f(x, y):
        return 2*x, 2*y

    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    U, V = grad_f(X, Y)

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, -U, -V, color='r')
    plt.contour(X, Y, f(X, Y), levels=10)

    # Optimization path
    path = []
    x_i, y_i = 4, 4
    path.append((x_i, y_i))
    learning_rate = 0.1
    for _ in range(10):
        grad_x, grad_y = grad_f(x_i, y_i)
        x_i -= learning_rate * grad_x
        y_i -= learning_rate * grad_y
        path.append((x_i, y_i))
    path = np.array(path)

    plt.plot(path[:, 0], path[:, 1], 'o-', color='b', label='Optimization Path')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vector Field with Optimization Path")
    plt.legend()
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, save_path):
    ensure_dir(save_path)
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # Binarize the output
    y_true_binary = (y_true > np.mean(y_true)).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    average_precision = average_precision_score(y_true_binary, y_scores)

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'AP = {average_precision:0.2f}')

    # Add iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve with Iso-F1 Lines")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate more plots")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    config = load_config(args.config)

    save_dir = Path("C:\\Users\\yasin\\Documents\\projectv10\\projectv10\\output_plt")
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = ThreeChargeDataset(
        num_samples=config['samples'],
        position_range=config.get('position_range', 10.0),
        charge_range=config.get('charge_range', 5.0),
        seed=config.get('seed', 42)
    )
    
    X, y = dataset.X, dataset.y
    feature_names = dataset.feature_names()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.get('seed', 42))

    # --- Train models ---
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=config.get('seed', 42)),
        "Random Forest": RandomForestRegressor(random_state=config.get('seed', 42)),
        "Linear Regression": LinearRegression()
    }

    metrics = {
        "R2 Score": [],
        "MAE": []
    }
    model_names = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics["R2 Score"].append(r2_score(y_test, y_pred))
        metrics["MAE"].append(mean_absolute_error(y_test, y_pred))
        model_names.append(name)

    # --- Generate plots ---

    # Hexbin density plot of two features
    plot_hexbin_density(X_train[:, 0], X_train[:, 1], save_dir / "hexbin_density_r1x_r1y.png")

    # ECDF of the first target variable
    plot_ecdf_with_ci(y_train[:, 0], save_dir / "ecdf_q1.png")

    # Violin plot of residuals
    residuals = y_test - models["Random Forest"].predict(X_test)
    plot_violin_with_medians(residuals, save_dir / "residuals_violin.png")

    # Waterfall chart
    plot_waterfall(feature_names, models["Random Forest"].feature_importances_, save_dir / "waterfall_chart.png")

    # Contour map
    plot_contour_map(X_train[:, 0], X_train[:, 1], y_train[:, 0], save_dir / "contour_map_q1.png")

    # Pareto Frontier (example with feature importances)
    costs = np.random.rand(len(feature_names))
    benefits = models["Random Forest"].feature_importances_
    plot_pareto_frontier(costs, benefits, save_dir / "pareto_frontier_features.png")

    # Scatter with covariance ellipse
    plot_scatter_with_covariance_ellipse(X_train[:, 0], X_train[:, 1], save_dir / "scatter_covariance_r1x_r1y.png")

    # Fan chart
    n_boots = 20
    y_preds = np.zeros((n_boots, len(X_test)))
    for i in range(n_boots):
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot, y_boot = X_train[indices], y_train[indices]
        boot_model = DecisionTreeRegressor(random_state=i)
        boot_model.fit(X_boot, y_boot)
        y_preds[i] = boot_model.predict(X_test)[:, 0]

    sorted_indices = np.argsort(X_test[:, 0])
    plot_fan_chart(X_test[sorted_indices, 0], y_preds[:, sorted_indices], save_dir / "fan_chart_q1.png")

    # Phase map
    plot_phase_map(DecisionTreeRegressor(), X_train, y_train[:, 0], save_dir / "phase_map_q1.png")

    # Radar chart
    plot_radar_chart(metrics, model_names, save_dir / "radar_chart_model_comparison.png")

    # Vector field
    plot_vector_field(save_dir / "vector_field_example.png")

    # Precision-Recall curve
    y_scores = models["Random Forest"].predict(X_test)[:, 0]
    plot_precision_recall_curve(y_test[:, 0], y_scores, save_dir / "precision_recall_q1.png")

    print(f"Generated plots in {save_dir}")

if __name__ == '__main__':
    main()