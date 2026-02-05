import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Metrik Hesaplama Fonksiyonu
def calculate_metrics(y_true, y_pred):
    """Verilen gerçek ve tahmin edilen değerler için metrikleri hesaplar."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # MAE de artık return ediliyor
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}


def run_linear_regression(data_path="../data/processed_data/"):
    """
    Lineer Regresyon modelini eğitir, Doğrulama/Test setlerinde değerlendirir
    ve sonuçları kaydeder.
    """
    print("--- Lineer Regresyon Modeli Başlatılıyor ---")

    # VERİ YÜKLEME
    try:
        X_train = pd.read_pickle(os.path.join(data_path, "X_train.pkl"))
        X_val = pd.read_pickle(os.path.join(data_path, "X_val.pkl"))
        X_test = pd.read_pickle(os.path.join(data_path, "X_test.pkl"))
        y_train = pd.read_pickle(os.path.join(data_path, "y_train.pkl"))
        y_val = pd.read_pickle(os.path.join(data_path, "y_val.pkl"))
        y_test = pd.read_pickle(os.path.join(data_path, "y_test.pkl"))
        print("Veriler başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"HATA: Veri dosyaları '{data_path}' yolunda bulunamadı. Lütfen yolu kontrol edin.")
        return  # Fonksiyondan çık

    # MODEL EĞİTİMİ VE SÜRE ÖLÇÜMÜ
    start_time = time.time()

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nEğitim Süresi: {training_time:.4f} saniye")

    # PERFORMANS DEĞERLENDİRMESİ

    # 1. Doğrulama Seti (Validation) Performansı
    y_val_pred = model_lr.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)

    print("\n--- Doğrulama (Validation) Metrikleri ---")
    print(f"R²: {val_metrics['R2']:.4f}")
    # RMSE ve MAE okunaklı formatta
    print(f"RMSE: {val_metrics['RMSE']:,.2f}") 
    print(f"MAE: {val_metrics['MAE']:,.2f}")

    # 2. Test Seti Performansı
    y_test_pred = model_lr.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print("\n--- Test Metrikleri ---")
    print(f"R²: {test_metrics['R2']:.4f}")
    # RMSE ve MAE okunaklı formatta
    print(f"RMSE: {test_metrics['RMSE']:,.2f}") 
    print(f"MAE: {test_metrics['MAE']:,.2f}")

    # Cross-Validation Sonuçları (Train seti üzerinde)
    cv_scores = cross_val_score(model_lr, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_r2_mean = np.mean(cv_scores)
    print(f"\n5-Fold Cross-Validation R² Ortalaması: {cv_r2_mean:.4f}")

    # ÖZELLİK ÖNEMİ YORUMLANMASI
    print("\n--- Özellik Önemleri (Katsayıların Mutlak Değeri) ---")
    coefficients = pd.Series(model_lr.coef_, index=X_train.columns)
    print(coefficients.abs().sort_values(ascending=False).head(5).to_string())

    # MODEL VE SONUÇLARI KAYDETME

    MODEL_DIR = 'models'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    results_lr = {
        'training_time': training_time,
        'validation': val_metrics,
        'test': test_metrics,
        'cv_r2_mean': cv_r2_mean
    }

    # Modeli ve sonuçları klasöre kaydet
    joblib.dump(model_lr, os.path.join(MODEL_DIR, 'model_lr.pkl'))
    joblib.dump(results_lr, os.path.join(MODEL_DIR, 'results_lr.pkl'))  # Tüm metrikler ve süre buraya kaydedildi.

    print(f"\n[INFO] Lineer Regresyon modeli ve sonuçları '{MODEL_DIR}' klasörüne başarıyla kaydedildi.")
    print("--- Lineer Regresyon Modeli Tamamlandı ---")


if __name__ == "__main__":
    run_linear_regression()