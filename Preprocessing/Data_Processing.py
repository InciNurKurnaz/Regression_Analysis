import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Kayıt klasörünü tanımla
OUTPUT_DIR = '../data/processed_data'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Kütüphaneler başarıyla yüklendi. Kayıt klasörü:", OUTPUT_DIR)
start_time = time.time()

# 1. VERİ YÜKLEME

print("\n Veri Yükleniyor...")
file_path = '../data/housing.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"HATA: Dosya '{file_path}' bulunamadı. Lütfen yolu kontrol edin.")
    exit()

# Sütun isimlerini netleştirme
df.columns = [
    'Longitude', 'Latitude', 'Housing_Median_Age', 'Total_Rooms',
    'Total_Bedrooms', 'Population', 'Households', 'Median_Income',
    'Median_House_Value', 'Ocean_Proximity'
]

# Hedef değişken (Y) ve Bağımsız değişkenler (X)
X = df.drop('Median_House_Value', axis=1)
y = df['Median_House_Value']

print(f"Veri Seti Boyutu: {df.shape}")
print(f"Toplam Örnek Sayısı: {len(df)}")


# 2. VERİ ÖN İŞLEME AŞAMASI

print("\nBölüm 2: Eksik Veri Doldurma ve Özellik Dönüşümü")

# Kategorik ve Sayısal Sütunları Tanımlama
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = ['Ocean_Proximity']
print(f"Ham Sayısal Özellik Sayısı: {len(numerical_features)}")
print(f"Ham Kategorik Özellik Sayısı: {len(categorical_features)}")

# Özellik Dönüşümleri için Pipeline Hazırlama
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Tüm dönüşümleri birleştirme
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
], remainder='passthrough')

# Veri setine dönüştürmeyi uygula
X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

print("\n--- Özellik Dönüşümü Sonucu ---")
print(f"One-Hot Encoding Sonrası Toplam Özellik Sayısı (Aykırı Değer Öncesi): {X_processed_df.shape[1]}")


# 3. AYKIRI DEĞER (OUTLIER) ANALİZİ VE İŞLENMESİ

print("\nBölüm 3: Aykırı Değer Analizi ve İşlenmesi")

# Hedef değişkende (y) aykırı değerleri tespit etme
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

outlier_indices = y[y > upper_bound].index
outlier_count = len(outlier_indices)
initial_size = len(X)

print(f"Tespit Edilen Aykırı Değer Sayısı (Hedef Değişken > Üst Sınır): {outlier_count}")
print(f"Aykırı Değer Oranı: {(outlier_count / initial_size) * 100:.2f}%")

# Aykırı değerleri veri setinden çıkarma
X_clean = X_processed_df.drop(outlier_indices)
y_clean = y.drop(outlier_indices)

print(f"Aykırı Değerler Çıkarıldıktan Sonra Kalan Örnek Sayısı: {len(X_clean)}")


# 4. ÖZELLİK SEÇİMİ (Feature Selection) - Korelasyon Analizi

print("\nBölüm 4: Özellik Seçimi (Filtre Yöntemi)")

# Bağımsız değişkenleri ve hedef değişkeni birleştirerek korelasyon hesapla
df_corr = pd.concat([X_clean, y_clean], axis=1)
correlations = df_corr.corr()['Median_House_Value'].abs().sort_values(ascending=False)
correlations_features = correlations.drop('Median_House_Value')

print("--- Hedef Değişken İle İlk 5 Korelasyonu Yüksek Özellik ---")
print(correlations_features.head(5).to_string())

# Korelasyonu düşük olan özelliklerin listesini çıkar (Eşik: |r| < 0.05)
low_corr_features = correlations_features[correlations_features < 0.05].index.tolist()

# Özellik seçimini uygulama: Düşük korelasyonlu özellikleri modelden çıkar
X_selected = X_clean.drop(columns=low_corr_features)

print(f"\nKorelasyon Eşiği (< 0.05) Altında Kalan Özellik Sayısı: {len(low_corr_features)}")
print(f"Çıkarılan Özellikler: {low_corr_features}")
print(f"Ön İşleme ve Özellik Seçimi Sonrası Nihai Özellik Sayısı: {X_selected.shape[1]}")


# 5. VERİ SETİNİN BÖLÜNMESİ (TRAIN-VALIDATION-TEST: 70-15-15)

print("\nBölüm 5: Veri Setinin Train-Validation-Test Olarak Ayrılması")

# 5.1. Önce Test setini ayır (%15)
test_size_split = 0.15
X_temp, X_test, y_temp, y_test = train_test_split(
    X_selected, y_clean, test_size=test_size_split, random_state=42
)

# 5.2. Kalan X_temp (%85) içinden Validation setini ayır (%15 / %85 ≈ %17.65)
validation_split = 0.15 / (1 - 0.15)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=validation_split, random_state=42
)

# Kontrol ve Çıktılar
total_samples = len(X_clean)
print(f"\n--- Sonuç: Veri Seti Oranları ---")
print(f"Eğitim Seti (Train) Boyutu: {X_train.shape} | Oran: {len(X_train)/total_samples:.2%} (%%70.00)")
print(f"Doğrulama Seti (Validation) Boyutu: {X_val.shape} | Oran: {len(X_val)/total_samples:.2%} (%%15.00)")
print(f"Test Seti (Test) Boyutu: {X_test.shape} | Oran: {len(X_test)/total_samples:.2%} (%%15.00)")


# 6. VERİLERİ KAYDETME

# Verileri belirlenen klasöre kaydet
X_train.to_pickle(os.path.join(OUTPUT_DIR, "X_train.pkl"))
X_val.to_pickle(os.path.join(OUTPUT_DIR, "X_val.pkl"))
X_test.to_pickle(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_train.to_pickle(os.path.join(OUTPUT_DIR, "y_train.pkl"))
y_val.to_pickle(os.path.join(OUTPUT_DIR, "y_val.pkl"))
y_test.to_pickle(os.path.join(OUTPUT_DIR, "y_test.pkl"))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n[INFO] Ön İşleme Süresi: {elapsed_time:.4f} saniye")
print(f"\nÖn İşleme tamamlandı ve veriler '{OUTPUT_DIR}' klasörüne kaydedildi.")