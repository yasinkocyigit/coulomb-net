
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

from src.dataset import ThreeChargeDataset
from src.data import set_seed

# Helper to check for XGBoost
def try_get_xgb():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError:
        return None

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_models(seed=0):
    models = {
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 7)),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=seed),
        "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed), n_jobs=-1)
    }
    XGB = try_get_xgb()
    if XGB:
        models["XGBoost"] = MultiOutputRegressor(XGB(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed, n_jobs=-1), n_jobs=-1)
    else:
        print("XGBoost not found, skipping.")
    return models

def main():
    parser = argparse.ArgumentParser(description="Generate and Analyze Model Plots")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    config = load_config(args.config)

    # Setup directories and seed
    output_dir = Path("C:/Users/yasin/Documents/projectv10/projectv10/output_plt2")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.get('seed', 42)
    set_seed(seed)

    # Load Data
    dataset = ThreeChargeDataset(
        num_samples=config['samples'],
        position_range=config.get('position_range', 10.0),
        charge_range=config.get('charge_range', 5.0),
        seed=seed
    )
    X, y = dataset.X, dataset.y
    feature_names = dataset.feature_names()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Train Models
    models = get_models(seed)
    results = {}
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "r2_mean": r2_score(y_test, y_pred),
            "mse_mean": mean_squared_error(y_test, y_pred)
        }
        trained_models[name] = model

    # Start analysis report
    analysis_report = "## MODELLER İÇİN GRAFİK ANALİZ RAPORU ##\n\n"
    
    # 1. Model Comparison Plot
    print("Generating Model Comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df_results = pd.DataFrame(results).T
    df_results.sort_values('r2_mean', ascending=True, inplace=True)
    axes[0].barh(df_results.index, df_results['r2_mean'])
    axes[0].set_title('Model Karşılaştırması: R² Skoru (Yüksek olan daha iyi)')
    axes[0].set_xlabel('Ortalama R² Skoru')

    df_results.sort_values('mse_mean', ascending=False, inplace=True)
    axes[1].barh(df_results.index, df_results['mse_mean'])
    axes[1].set_title('Model Karşılaştırması: Ortalama Karesel Hata (MSE) (Düşük olan daha iyi)')
    axes[1].set_xlabel('Ortalama MSE')
    
    plt.tight_layout()
    comp_path = output_dir / "1_model_comparison.png"
    plt.savefig(comp_path)
    plt.close()

    analysis_report += (
        "### 1. Model Karşılaştırma Grafiği\n\n"
        f"- **Grafik:** `{comp_path.name}`\n"
        "- **Açıklama:** Bu grafik, eğitilen tüm temel modellerin genel performansını iki ana metrik üzerinden karşılaştırır: R² skoru (modelin veriyi ne kadar iyi açıkladığı) ve MSE (tahminlerin ortalama hatası).\n"
        f"- **Çıkarım:** R² grafiğine göre en yüksek skoru alan model **{df_results.sort_values('r2_mean', ascending=False).index[0]}** olmuştur. Bu, veri setindeki değişkenliği en iyi açıklayan modelin bu olduğunu gösterir. Benzer şekilde, MSE grafiğindeki en düşük hata oranı da bu modelin en isabetli tahminleri yaptığını doğrulamaktadır. Model seçiminde ilk tercih bu model olmalıdır.\n\n"
    )

    best_model_name = df_results.sort_values('mse_mean', ascending=True).index[0]
    best_model = trained_models[best_model_name]

    # 2. Residuals vs. Predicted Plot (for each model)
    print("Generating Residuals plots...")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_pred.mean(axis=1), y=residuals.mean(axis=1), alpha=0.5, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Tahmin Edilen Değer (Ortalama)")
        ax.set_ylabel("Hata (Artık Değer)")
        ax.set_title(f'Hata Analizi Grafiği: {name}')
        res_path = output_dir / f"2_residuals_{name}.png"
        plt.savefig(res_path)
        plt.close()

        analysis_report += (
            f"### 2. Hata Analizi Grafiği ({name})\n\n"
            f"- **Grafik:** `{res_path.name}`\n"
            "- **Açıklama:** Bu grafik, modelin hatalarının (artık değerler) tahmin edilen değerlere karşı nasıl dağıldığını gösterir.\n"
            "- **Çıkarım:** İdeal bir modelde, noktaların sıfır çizgisi etrafında rastgele ve simetrik bir şekilde dağılması beklenir. Eğer bu grafikte bir desen (örneğin huni şekli veya bir eğri) görülüyorsa, bu durum modelin sistematik bir hata yaptığını gösterir. Bu grafiği inceleyerek, modelin belirli büyüklükteki tahminlerde daha fazla veya daha az hata yapma eğiliminde olup olmadığını anlayabiliriz.\n\n"
        )

    # 3. Feature Importance Plot
    print("Generating Feature Importance plot...")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'estimators_'): # For MultiOutput
        importances = np.mean([est.feature_importances_ for est in best_model.estimators_], axis=0)
    else:
        importances = None

    if importances is not None:
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=df_imp)
        plt.title(f'En Önemli 10 Özellik ({best_model_name})')
        fi_path = output_dir / "3_feature_importance.png"
        plt.savefig(fi_path)
        plt.close()

        top_features = df_imp['feature'].tolist()
        analysis_report += (
            "### 3. Özellik Önem Sıralaması Grafiği\n\n"
            f"- **Grafik:** `{fi_path.name}`\n"
            f"- **Açıklama:** Bu grafik, en iyi model olan **{best_model_name}**'in tahmin yaparken hangi girdi özelliklerini ne kadar dikkate aldığını gösterir.\n"
            f"- **Çıkarım:** Grafiğin en üstündeki özellikler, modelin kararlarını en çok etkileyenlerdir. Örneğin, `{top_features[0]}` ve `{top_features[1]}` gibi özelliklerin en önemli olması, bu fiziksel büyüklüklerin modelin tahmin doğruluğu için kritik olduğunu ortaya koyar. Bu bilgi, hangi verilerin daha değerli olduğunu anlamamıza yardımcı olur.\n\n"
        )

    # 4. Partial Dependence Plots
    print("Generating Partial Dependence plots...")
    if importances is not None:
        top_2_features_indices = [feature_names.index(f) for f in top_features[:2]]
        # Create PDP for each target output (q1 and q2)
        for i, target_name in enumerate(['q1', 'q2']):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            PartialDependenceDisplay.from_estimator(
                best_model, X_train, features=top_2_features_indices, 
                feature_names=feature_names, ax=axes, grid_resolution=50, target=i
            )
            plt.suptitle(f'Kısmi Bağımlılık Grafikleri ({best_model_name}) - Hedef: {target_name}')
            pdp_path = output_dir / f"4_partial_dependence_plots_{target_name}.png"
            plt.savefig(pdp_path)
            plt.close()

            analysis_report += (
                f"### 4. Kısmi Bağımlılık Grafikleri (PDP) - Hedef {target_name}\n\n"
                f"- **Grafik:** `{pdp_path.name}`\n"
                f"- **Açıklama:** Bu grafikler, en önemli iki özelliğin ({top_features[0]} ve {top_features[1]}) değerleri değiştikçe, modelin **{target_name}** tahminlerinin (diğer tüm özellikler sabit tutulduğunda) nasıl değiştiğini gösterir.\n"
                "- **Çıkarım:** Her bir grafikteki eğrinin şekli, özellik ile hedef arasındaki ilişkiyi ortaya koyar. Örneğin, bir özellik arttıkça tahmin de sürekli artıyorsa aralarında pozitif bir ilişki vardır. Eğrinin düz olduğu bölgeler, özelliğin o aralıkta tahmini etkilemediğini gösterir. Bu, modelin 'karar verme mantığını' anlamak için çok güçlü bir araçtır.\n\n"
            )

    # 5. Actual vs. Predicted Plot
    print("Generating Actual vs. Predicted plot...")
    y_pred_best = best_model.predict(X_test)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test.mean(axis=1), y=y_pred_best.mean(axis=1), alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.title(f'Gerçek vs. Tahmin Grafiği ({best_model_name})')
    avp_path = output_dir / "5_actual_vs_predicted.png"
    plt.savefig(avp_path)
    plt.close()

    analysis_report += (
        "### 5. Gerçek Değerler vs. Tahmin Edilen Değerler Grafiği\n\n"
        f"- **Grafik:** `{avp_path.name}`\n"
        f"- **Açıklama:** Bu grafik, en iyi model olan **{best_model_name}**'in tahminlerinin (Y ekseni) gerçek değerlere (X ekseni) ne kadar yakın olduğunu gösterir.\n"
        "- **Çıkarım:** Mükemmel bir modelde, tüm noktaların kırmızı kesikli çizginin (y=x doğrusu) üzerinde yer alması beklenir. Noktaların bu çizgi etrafında ne kadar sıkı bir şekilde kümelendiği, modelin genel doğruluğunun bir göstergesidir. Bu grafikteki dağılım, modelin tahminlerinin ne kadar isabetli olduğunu görsel olarak teyit eder.\n\n"
    )

    # Write analysis report to file
    report_path = output_dir / "plot_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(analysis_report)

    print(f"\nSuccessfully generated plots and analysis report in: {output_dir}")

if __name__ == '__main__':
    main()
