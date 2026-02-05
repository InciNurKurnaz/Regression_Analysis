import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Metrik Hesaplama Fonksiyonu
def calculate_metrics(y_true, y_pred):
    """Verilen gerçek ve tahmin edilen değerler için metrikleri hesaplar."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}


def run_gradient_boosting(data_path="../data/processed_data/"):
    """
    Gradient Boosting Regressor modelini eğitir, değerlendirir ve sonuçları kaydeder.
    """
    print("--- Gradient Boosting Regressor Modeli Başlatılıyor ---")

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
        return

    # MODEL EĞİTİMİ VE SÜRE ÖLÇÜMÜ
    start_time = time.time()

    # Not: Parametreler (n_estimators=100, learning_rate=0.1, max_depth=3) varsayılan olarak seçilmiştir.
    model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_gbr.fit(X_train, y_train)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nEğitim Süresi: {training_time:.4f} saniye")


    # PERFORMANS DEĞERLENDİRMESİ

    # 1. Doğrulama Seti (Validation) Performansı
    y_val_pred = model_gbr.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)

    print("\n--- Doğrulama (Validation) Metrikleri ---")
    print(f"R²: {val_metrics['R2']:.4f}")
    print(f"RMSE: {val_metrics['RMSE']:,.2f}")  # Okunabilir format
    print(f"MAE: {val_metrics['MAE']:,.2f}")  # MAE

    # 2. Test Seti Performansı
    y_test_pred = model_gbr.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print("\n--- Test Metrikleri ---")
    print(f"R²: {test_metrics['R2']:.4f}")
    print(f"RMSE: {test_metrics['RMSE']:,.2f}")  # Okunabilir format
    print(f"MAE: {test_metrics['MAE']:,.2f}")  # MAE

    # Cross-Validation Sonuçları (Train seti üzerinde)
    cv_scores = cross_val_score(model_gbr, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_r2_mean = np.mean(cv_scores)
    print(f"\n5-Fold Cross-Validation R² Ortalaması: {cv_r2_mean:.4f}")


    # ÖZELLİK ÖNEMİ YORUMLANMASI (Feature Importance)

    print("\n--- Özellik Önemleri (Feature Importance) ---")
    feature_importances = pd.Series(model_gbr.feature_importances_, index=X_train.columns)
    print(feature_importances.sort_values(ascending=False).head(5).to_string())


    # MODEL VE SONUÇLARI KAYDETME

    MODEL_DIR = 'models'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    results_gbr = {
        'training_time': training_time,
        'validation': val_metrics,
        'test': test_metrics,
        'cv_r2_mean': cv_r2_mean,
        'feature_importances': feature_importances.sort_values(ascending=False).to_dict()
    }

    # Modeli ve sonuçları modelling/models klasörüne kaydet
    joblib.dump(model_gbr, os.path.join(MODEL_DIR, 'model_gbr.pkl'))
    joblib.dump(results_gbr, os.path.join(MODEL_DIR, 'results_gbr.pkl'))

    print(f"\n[INFO] Gradient Boosting modeli ve sonuçları '{MODEL_DIR}' klasörüne başarıyla kaydedildi.")
    print("--- Gradient Boosting Regressor Modeli Tamamlandı ---")


if __name__ == "__main__":
    run_gradient_boosting()