# Dengesiz Veri Setlerinde Doğru Metrik Seçimi: Accuracy Tuzağı, Precision, Recall ve F1

> **Bağlantılı olduğu ders:** Ders 5 – Müşteri Segmentasyonu ve Churn Tahmini
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok)

## 1. Problem: %95 doğruluk neden kötü bir model olabilir?

Churn (müşteri kaybı) tahmininde sınıflar genellikle **dengesizdir**: örneğin 1000 müşteriden yalnızca 50'si (%5) ayrılır. Bu durumda **"hiç kimse ayrılmayacak"** diyen ve hiçbir şey öğrenmemiş bir model bile **%95 accuracy** elde eder — ama ayrılacak müşterilerin **hiçbirini** yakalayamaz.

Bu yüzden dengesiz veride tek başına *accuracy* (doğruluk) yanıltıcıdır. Doğru karar için **karışıklık matrisine** ve ondan türetilen metriklere bakmak gerekir.

## 2. Karışıklık Matrisi (Confusion Matrix)

|                      | Tahmin: Churn (1) | Tahmin: Kalıyor (0) |
|----------------------|:-----------------:|:-------------------:|
| **Gerçek: Churn (1)**   | TP (Doğru Pozitif) | FN (Yanlış Negatif – *kaçırılan*) |
| **Gerçek: Kalıyor (0)** | FP (Yanlış Pozitif – *yanlış alarm*) | TN (Doğru Negatif) |

## 3. Temel Metrikler

| Metrik | Formül | Sorduğu soru |
|---|---|---|
| **Accuracy** | (TP + TN) / Toplam | Tüm tahminlerin ne kadarı doğru? |
| **Precision** (Kesinlik) | TP / (TP + FP) | "Churn" dediklerimin ne kadarı gerçekten churn? |
| **Recall** (Duyarlılık) | TP / (TP + FN) | Gerçek churn'lerin ne kadarını yakaladım? |
| **F1 Skoru** | 2 · P · R / (P + R) | Precision ve recall'un harmonik ortalaması |

- **Precision** yüksekse yanlış alarm azdır (kampanya bütçesi boşa gitmez).
- **Recall** yüksekse kaçırılan müşteri azdır.
- İkisi genellikle **ters yönde** hareket eder: karar eşiğini düşürürsek recall artar, precision düşer.

## 4. Uygulama: Sıfırdan Hesaplama ve Eşik Etkisi

Aşağıdaki kod yalnızca standart Python ile çalışır. Sentetik olarak 1000 müşterilik, %5 churn oranına sahip bir veri ve bir modelin ürettiği olasılık skorları oluşturulur.

```python
import random

random.seed(42)

# 1000 müşteri: %5'i churn (1), %95'i kalıyor (0)
y_gercek = [1] * 50 + [0] * 950

# Basit bir modelin ürettiği "churn olasılıkları" (sentetik):
# churn edenlerde skorlar daha yüksek ama sınıflar kısmen örtüşüyor.
skorlar = [min(1, max(0, random.gauss(0.60, 0.20))) for _ in range(50)] + \
          [min(1, max(0, random.gauss(0.25, 0.15))) for _ in range(950)]


def karisiklik_matrisi(y, y_tahmin):
    tp = sum(1 for g, t in zip(y, y_tahmin) if g == 1 and t == 1)
    fp = sum(1 for g, t in zip(y, y_tahmin) if g == 0 and t == 1)
    fn = sum(1 for g, t in zip(y, y_tahmin) if g == 1 and t == 0)
    tn = sum(1 for g, t in zip(y, y_tahmin) if g == 0 and t == 0)
    return tp, fp, fn, tn


def metrikler(y, y_tahmin):
    tp, fp, fn, tn = karisiklik_matrisi(y, y_tahmin)
    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


# 1) "Herkes kalacak" diyen tembel model
tembel = [0] * len(y_gercek)
print("Tembel model  -> acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f" % metrikler(y_gercek, tembel))

# 2) Skor tabanlı model, farklı eşiklerle
for esik in (0.3, 0.4, 0.5, 0.6):
    tahmin = [1 if s >= esik else 0 for s in skorlar]
    tp, fp, fn, tn = karisiklik_matrisi(y_gercek, tahmin)
    acc, prec, rec, f1 = metrikler(y_gercek, tahmin)
    print(f"Eşik={esik:.1f}  TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d}  "
          f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

# 3) İş maliyetiyle eşik seçimi:
# Kaçırılan bir churn (FN) 10 birim, gereksiz kampanya (FP) 1 birim maliyetli olsun.
print()
for esik in (0.3, 0.4, 0.5, 0.6):
    tahmin = [1 if s >= esik else 0 for s in skorlar]
    tp, fp, fn, tn = karisiklik_matrisi(y_gercek, tahmin)
    print(f"Eşik={esik:.1f}  maliyet = 10*{fn} + 1*{fp} = {10 * fn + fp}")
```

### Çıktı

```text
Tembel model  -> acc=0.950  prec=0.000  rec=0.000  f1=0.000
Eşik=0.3  TP= 48 FP=341 FN=  2 TN=609  acc=0.657  prec=0.123  rec=0.960  f1=0.219
Eşik=0.4  TP= 42 FP=151 FN=  8 TN=799  acc=0.841  prec=0.218  rec=0.840  f1=0.346
Eşik=0.5  TP= 37 FP= 58 FN= 13 TN=892  acc=0.929  prec=0.389  rec=0.740  f1=0.510
Eşik=0.6  TP= 26 FP=  8 FN= 24 TN=942  acc=0.968  prec=0.765  rec=0.520  f1=0.619

Eşik=0.3  maliyet = 10*2 + 1*341 = 361
Eşik=0.4  maliyet = 10*8 + 1*151 = 231
Eşik=0.5  maliyet = 10*13 + 1*58 = 188
Eşik=0.6  maliyet = 10*24 + 1*8 = 248
```

## 5. Sonuçların Yorumu

1. **Tembel model** %95 accuracy alıyor ama recall = 0. Tek bir churn müşterisini bile yakalamıyor → **işe yaramaz**.
2. **Eşik 0.5'te** skor tabanlı modelin accuracy'si (0.929) tembel modelden *düşük* görünüyor, ama 50 churn'ün 37'sini yakalıyor. Accuracy'ye bakıp bu modeli reddetmek büyük hata olurdu.
3. **Eşik düştükçe** recall artıyor (0.52 → 0.96), precision düşüyor (0.77 → 0.12). Bu, precision–recall dengesinin (trade-off) somut hâlidir.
4. **F1'e göre** en iyi eşik 0.6; **iş maliyetine göre** ise 0.5 (188 birim). Yani "en iyi metrik" problemin bağlamına göre değişir: kaçırılan bir müşteri, gereksiz bir indirim kuponundan çok daha pahalıysa recall'a ağırlık verilmelidir.

## 6. Sık Yapılan Hatalar

- ❌ Dengesiz veride yalnızca **accuracy** raporlamak.
- ❌ Varsayılan **0.5 eşiğini** sorgulamadan kullanmak. Eşik, bir *iş kararıdır*; doğrulama (validation) verisi üzerinde seçilmelidir — test verisi üzerinde değil.
- ❌ Train/test ayırırken sınıf oranını korumamak. `train_test_split(..., stratify=y)` kullanılmalıdır.
- ❌ Precision ve recall'u hangi sınıf için hesapladığını belirtmemek (pozitif sınıf genellikle azınlık sınıfıdır).
- ❌ Dengesizliği çözmek için oversampling (ör. SMOTE) uygulayıp bunu **bölmeden önce** tüm veriye yapmak → veri sızıntısı (data leakage).

## 7. scikit-learn ile Kısa Yol

Aynı hesaplar gerçek projelerde tek satırla yapılabilir:

```python
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

print(confusion_matrix(y_gercek, tahmin))
print(classification_report(y_gercek, tahmin, digits=3))

# Tüm eşikler için precision/recall değerleri:
precision, recall, esikler = precision_recall_curve(y_gercek, skorlar)
```

Dengesiz veride model karşılaştırmak için ROC-AUC yerine çoğu zaman **PR-AUC** (`average_precision_score`) daha bilgilendiricidir, çünkü çok sayıdaki TN'den etkilenmez.

## 8. Mini Alıştırma

**Soru:** Kaçırılan churn'ün maliyeti 10 değil **3 birim** olsaydı (FP yine 1 birim), yukarıdaki dört eşikten hangisi en düşük maliyeti verirdi?

<details>
<summary>Çözüm</summary>

Maliyet = 3 · FN + 1 · FP

| Eşik | Hesap | Maliyet |
|---|---|---|
| 0.3 | 3·2 + 341 | 347 |
| 0.4 | 3·8 + 151 | 175 |
| 0.5 | 3·13 + 58 | 97 |
| 0.6 | 3·24 + 8 | **80** |

Kaçırmanın maliyeti azaldığında, yanlış alarmları azaltan **daha yüksek eşik (0.6)** daha karlı hâle gelir.
</details>

---
*Özet:* Dengesiz veride önce karışıklık matrisine bak, precision ve recall'u birlikte değerlendir, karar eşiğini de iş maliyetine göre seç.
