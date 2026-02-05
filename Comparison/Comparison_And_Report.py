import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


# Modellerin ve sonuçların bulunduğu klasör yolu
MODEL_DIR = '../Modelling/models'

# Grafiklerin kaydedileceği klasör yolu
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# VERİ VE SONUÇLARI YÜKLEME

try:
    # X_test ve y_test'i processed_data klasöründen yükle
    X_test = pd.read_pickle("../data/processed_data/X_test.pkl")
    y_test = pd.read_pickle("../data/processed_data/y_test.pkl")

    results_lr = joblib.load(os.path.join(MODEL_DIR, 'results_lr.pkl'))
    results_rf = joblib.load(os.path.join(MODEL_DIR, 'results_rf.pkl'))
    results_gbr = joblib.load(os.path.join(MODEL_DIR, 'results_gbr.pkl'))

    model_lr = joblib.load(os.path.join(MODEL_DIR, 'model_lr.pkl'))
    model_rf = joblib.load(os.path.join(MODEL_DIR, 'model_rf.pkl'))
    model_gbr = joblib.load(os.path.join(MODEL_DIR, 'model_gbr.pkl'))

except FileNotFoundError as e:
    print(f"HATA: Dosya yüklenemedi. Lütfen 'models' ve 'processed_data' klasör yollarını kontrol edin. Hata: {e}")
    exit()

# SONUÇLARI BİRLEŞTİRME VE KARŞILAŞTIRMA

# Metrikleri ve Süreleri Toplama
comparison_data = {
    'Lineer Regresyon (LR)': {
        'R²': results_lr['test']['R2'],
        'RMSE': results_lr['test']['RMSE'],
        'MAE': results_lr['test']['MAE'],
        'Eğitim Süresi (s)': results_lr['training_time']
    },
    'Random Forest (RF)': {
        'R²': results_rf['test']['R2'],
        'RMSE': results_rf['test']['RMSE'],
        'MAE': results_rf['test']['MAE'],
        'Eğitim Süresi (s)': results_rf['training_time']
    },
    'Gradient Boosting (GBR)': {
        'R²': results_gbr['test']['R2'],
        'RMSE': results_gbr['test']['RMSE'],
        'MAE': results_gbr['test']['MAE'],
        'Eğitim Süresi (s)': results_gbr['training_time']
    }
}

comparison_df = pd.DataFrame(comparison_data).T
comparison_df = comparison_df.sort_values(by='R²', ascending=False)

print("--- MODELLERİN PERFORMANS KARŞILAŞTIRMASI ---")
print(comparison_df.applymap(lambda x: f'{x:,.4f}' if isinstance(x, (int, float)) else x))


# GRAFİK OLUŞTURMA VE KAYDETME

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Metriklerin Görsel Karşılaştırması (R², RMSE, MAE)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Regresyon Modelleri Performans Metrikleri Karşılaştırması (Test Seti)', fontsize=16)

metrics_to_plot = ['R²', 'RMSE', 'MAE']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, metric in enumerate(metrics_to_plot):
    data = comparison_df[metric]
    bars = axes[i].bar(data.index, data.values, color=colors[i])

    if metric == 'R²':
        axes[i].set_ylim(0.55, 0.70)
        axes[i].set_title(f'{metric} (Yüksek İyidir)')
    else:
        axes[i].set_title(f'{metric} (Düşük İyidir)')

    axes[i].tick_params(axis='x', rotation=45)
    axes[i].bar_label(bars, fmt='%.4f')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison_bar.png'))
plt.close(fig)  # Hafızayı temizle
print(f"\n[INFO] Metrik karşılaştırma grafiği '{OUTPUT_DIR}/metrics_comparison_bar.png' olarak kaydedildi.")



# 2. Eğitim Süresi Karşılaştırması
fig, ax = plt.subplots(figsize=(8, 6))
data = comparison_df['Eğitim Süresi (s)']
bars = ax.bar(data.index, data.values, color='#9467bd')
ax.set_title('Modellerin Eğitim Süresi Karşılaştırması (Saniye)')
ax.set_ylabel('Eğitim Süresi (s)')
ax.tick_params(axis='x', rotation=45)
ax.bar_label(bars, fmt='%.4f')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_time_bar.png'))
plt.close(fig)  # Hafızayı temizle
print(f"[INFO] Eğitim süresi grafiği '{OUTPUT_DIR}/training_time_bar.png' olarak kaydedildi.")


print("\nKarşılaştırma ve raporlama tamamlandı.")