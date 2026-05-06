

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import sys

# Install SHAP library
try:
    import shap
except ImportError:
    print("SHAP library not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.dataset import ThreeChargeDataset
from src.data import set_seed

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate More Analyzed Model Plots")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    config = load_config(args.config)

    # Setup directories and seed
    output_dir = Path("C:/Users/yasin/Documents/projectv10/projectv10/output_plt2")
    report_path = output_dir / "plot_analysis_report.txt"
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

    # Train the best model (XGBoost)
    print("Training the best model (XGBoost)...")
    best_model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed, n_jobs=-1), n_jobs=-1)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred

    # Start appending to the analysis report
    analysis_report = "\n\n-- EK ANALİZLER --\n"

    # 6. Learning Curves
    print("Generating Learning Curves...")
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(.1, 1.0, 5), scoring="r2"
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim Skoru")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Çapraz Doğrulama Skoru")
    plt.xlabel("Eğitim Örnek Sayısı")
    plt.ylabel("R² Skoru")
    plt.title('Öğrenme Eğrileri (Learning Curves)')
    plt.legend(loc="best")
    plt.grid()
    lc_path = output_dir / "6_learning_curves.png"
    plt.savefig(lc_path)
    plt.close()

    analysis_report += (
        "### 6. Öğrenme Eğrileri Grafiği\n\n"
        f"- **Grafik:** `{lc_path.name}`\n"
        "- **Açıklama:** Bu grafik, eğitim veri setinin boyutu arttıkça modelin hem eğitim verisi (kırmızı) hem de test verisi (yeşil) üzerindeki performansının nasıl değiştiğini gösterir.\n"
        "- **Çıkarım:** Yeşil ve kırmızı çizgiler birbirine yaklaşıyor ve yüksek bir R² skorunda düzleşiyorsa, model iyi bir genelleme yeteneğine sahiptir. Eğer aralarında büyük bir boşluk varsa bu 'ezberlemeye' (overfitting), eğer ikisi de düşük bir skorda düzleşiyorsa bu 'yetersiz öğrenmeye' (underfitting) işaret eder. Bu grafik, modelin daha fazla veriden fayda sağlayıp sağlamayacağını anlamak için kritiktir.\n\n"
    )

    # 7. Distribution of Residuals
    print("Generating Distribution of Residuals plot...")
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals.mean(axis=1), kde=True, bins=30)
    plt.xlabel("Ortalama Hata (Artık Değer)")
    plt.ylabel("Frekans")
    plt.title('Hata Dağılımı Grafiği')
    plt.axvline(0, color='red', linestyle='--')
    dr_path = output_dir / "7_distribution_of_residuals.png"
    plt.savefig(dr_path)
    plt.close()

    analysis_report += (
        "### 7. Hata Dağılımı Grafiği\n\n"
        f"- **Grafik:** `{dr_path.name}`\n"
        "- **Açıklama:** Bu histogram, modelin tahmin hatalarının genel dağılımını gösterir.\n"
        "- **Çıkarım:** İdeal bir durumda, bu dağılımın ortalaması sıfır olan (kırmızı kesikli çizgi üzerinde merkezlenmiş) ve çan eğrisine (normal dağılım) benzeyen bir yapıda olması beklenir. Eğer dağılım bir yöne doğru çarpıksa (skewed), bu modelin sürekli olarak pozitif veya negatif yönde yanlı tahminler yapma eğiliminde olduğunu gösterir.\n\n"
    )

    # 8. SHAP Summary Plot
    print("Generating SHAP Summary Plot...")
    # SHAP can be slow, so we use a subset of the data
    explainer = shap.TreeExplainer(best_model.estimators_[0]) # Explainer for q1
    shap_values = explainer(X_test[:500,:])
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X_test[:500,:], feature_names=feature_names, show=False)
    plt.title('SHAP Değerleri Özet Grafiği (q1 için)')
    shap_path = output_dir / "8_shap_summary_plot.png"
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()

    analysis_report += (
        "### 8. SHAP Değerleri Özet Grafiği\n\n"
        f"- **Grafik:** `{shap_path.name}`\n"
        "- **Açıklama:** Bu gelişmiş grafik, her bir özelliğin modelin tahminleri üzerindeki etkisini özetler. Her bir nokta, tek bir tahmin için bir özelliğin SHAP değerini temsil eder.\n"
        "- **Çıkarım:**\n"
        "  - **Önem:** Özellikler en önemliden en aza doğru yukarıdan aşağıya sıralanır.\n"
        "  - **Etki:** Noktaların x eksenindeki konumu, tahmini ne kadar artırdığını veya azalttığını gösterir. Sağa giden noktalar tahmini artırır, sola gidenler azaltır.\n"
        "  - **Değer:** Noktanın rengi, o özellik için değerin yüksek (kırmızı) mi yoksa düşük (mavi) mü olduğunu belirtir. Örneğin, `r_target_z` için kırmızı noktaların solda toplanması, bu özelliğin yüksek değerlerinin tahmini düşürme eğiliminde olduğunu gösterir.\n\n"
    )

    # Append to the report file
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(analysis_report)

    print(f"\nSuccessfully added new plots and analysis to: {output_dir}")

if __name__ == '__main__':
    main()
