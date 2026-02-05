# California Housing Price Prediction (Regression Analysis)

Bu proje, California konut veri setini kullanarak evlerin medyan değerlerini tahmin etmek amacıyla geliştirilmiştir. Veri yüklemeden model değerlendirmeye kadar uçtan uca bir makine öğrenmesi hattı (pipeline) uygulanmıştır.

---

## 🛠️ Veri Ön İşleme (Preprocessing)
Model başarısını artırmak için aşağıdaki teknik adımlar uygulanmıştır:
* **Eksik Veri Yönetimi:** `SimpleImputer` kullanılarak eksik sayısal değerler medyan ile tamamlandı.
* **Özellik Ölçeklendirme:** Sayısal veriler `StandardScaler` ile normalize edildi.
* **Kategorik Dönüştürme:** `Ocean_Proximity` değişkeni `OneHotEncoder` ile işlendi.
* **Aykırı Değer (Outlier) Analizi:** Hedef değişken (`Median_House_Value`) üzerindeki aşırı uç değerler IQR yöntemi ile temizlendi.
* **Özellik Seçimi:** Hedef değişken ile korelasyonu düşük olan (|r| < 0.05) özellikler elendi.

---

## 🤖 Kullanılan Modeller
Tahmin performansı üç farklı algoritma ile karşılaştırılmıştır:

1. **Linear Regression:** Baz model olarak kullanıldı.
2. **Random Forest Regressor:** Topluluk öğrenmesi ile yüksek doğruluk hedeflendi.
3. **Gradient Boosting Regressor:** Hata payını minimize etmek için uygulandı.

---

## 📊 Veri Seti Bölümlemesi
Modelin genelleme yeteneğini ölçmek için veri seti şu oranlarda ayrılmıştır:
* **Eğitim (Train):** %70
* **Doğrulama (Validation):** %15
* **Test:** %15
