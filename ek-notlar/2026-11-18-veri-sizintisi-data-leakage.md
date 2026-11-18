# Veri Sızıntısı (Data Leakage): Modeliniz Neden Gerçekte Olduğundan Daha İyi Görünüyor?

> **Bağlantılı olduğu dersler:** Ders 2 – Denetimli Öğrenme · Ders 3 – Keşifsel Veri Analizi (EDA) · Ders 5 – Churn Tahmini
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Tanım

**Veri sızıntısı**, modelin eğitim sırasında, gerçek kullanımda (tahmin anında) **erişemeyeceği bir bilgiyi** görmesidir. Sonuç hep aynıdır: Doğrulama/test skoru parlak çıkar, model canlıya alındığında performans çöker.

Sızıntı bir "kod hatası" gibi görünmez — program çalışır, hata mesajı yoktur, sadece **sayılar yalan söyler**. Bu yüzden en tehlikeli hatalardan biridir.

## 2. İki Ana Tür

| Tür | Ne oluyor? | Tipik örnek |
|---|---|---|
| **Hedef sızıntısı** (target leakage) | Bir özellik, hedefin *sonucu* olarak oluşuyor ya da hedeften sonra kaydediliyor | Churn tahmininde "iptal_talebi_tarihi" ya da "son ay iade edilen ücret" sütunu |
| **Eğitim–test bulaşması** (train–test contamination) | Ön işleme / seçim adımları test verisini de görerek yapılıyor | Ölçeklemeyi, eksik değer doldurmayı veya özellik seçimini veriyi bölmeden *önce* tüm veriyle yapmak |

Kendinize sorun: **"Bu bilgi, tahmin yapacağım anda elimde olacak mı?"** Cevap "hayır" ise o özellik ya da o işlem sızıntıdır.

## 3. Deney: Saf Gürültüden %90 Doğruluk

Aşağıdaki veride **hiçbir gerçek sinyal yok**: 60 örnek, 2000 tamamen rastgele özellik, etiketler yazı-tura ile belirleniyor. Dürüst bir modelin doğruluğu ≈ %50 olmalı.

İki yol karşılaştırılıyor:

- **YANLIŞ:** Etiketle en çok korele 20 özellik **tüm veriyle** seçiliyor, sonra 5 katlı çapraz doğrulama yapılıyor.
- **DOĞRU:** Özellik seçimi **her katın içinde, yalnızca o katın eğitim verisiyle** yapılıyor.

Model olarak basit bir *en yakın merkez* (nearest centroid) sınıflandırıcısı kullanılıyor.

```python
import random
import statistics

random.seed(42)

# 1) Tamamen rastgele veri: 60 örnek, 2000 özellik, etiketler yazı-tura.
#    Özelliklerle etiket arasında HİÇBİR gerçek ilişki yok → dürüst doğruluk ≈ %50.
n, p, secilecek = 60, 2000, 20
X = [[random.gauss(0, 1) for _ in range(p)] for _ in range(n)]
y = [random.randint(0, 1) for _ in range(n)]


def korelasyon(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    pay = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    payda = (sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b)) ** 0.5
    return pay / payda if payda else 0.0


def en_iyi_ozellikler(satirlar, etiketler, k):
    """Etiketle |korelasyonu| en yüksek k özelliğin indekslerini döndürür."""
    skorlar = []
    for j in range(len(satirlar[0])):
        sutun = [r[j] for r in satirlar]
        skorlar.append((abs(korelasyon(sutun, etiketler)), j))
    return [j for _, j in sorted(skorlar, reverse=True)[:k]]


def en_yakin_merkez_egit(satirlar, etiketler, ozellikler):
    merkez = {}
    for sinif in (0, 1):
        grup = [r for r, e in zip(satirlar, etiketler) if e == sinif]
        merkez[sinif] = [statistics.fmean(r[j] for r in grup) for j in ozellikler]
    return merkez


def tahmin(merkez, satir, ozellikler):
    x = [satir[j] for j in ozellikler]
    uzaklik = {s: sum((a - b) ** 2 for a, b in zip(x, m)) for s, m in merkez.items()}
    return min(uzaklik, key=uzaklik.get)


def k_katli_dogruluk(X, y, k=5, onceden_secilen=None):
    """onceden_secilen verilirse YANLIŞ yol (seçim tüm veriyle yapılmış),
    verilmezse DOĞRU yol (seçim her katta sadece eğitim verisiyle yapılır)."""
    indeksler = list(range(len(X)))
    random.Random(0).shuffle(indeksler)
    katlar = [indeksler[i::k] for i in range(k)]
    dogru = 0
    for test_idx in katlar:
        test_kume = set(test_idx)
        egitim_idx = [i for i in indeksler if i not in test_kume]
        Xe, ye = [X[i] for i in egitim_idx], [y[i] for i in egitim_idx]
        ozellikler = onceden_secilen or en_iyi_ozellikler(Xe, ye, secilecek)
        model = en_yakin_merkez_egit(Xe, ye, ozellikler)
        dogru += sum(tahmin(model, X[i], ozellikler) == y[i] for i in test_idx)
    return dogru / len(X)


# YANLIŞ: Özellik seçimi TÜM veriyle (test katları dahil) yapılıyor
sizdiran = en_iyi_ozellikler(X, y, secilecek)
print(f"Sızıntılı CV doğruluğu : {k_katli_dogruluk(X, y, onceden_secilen=sizdiran):.2f}")

# DOĞRU: Özellik seçimi her katın içinde, sadece eğitim verisiyle
print(f"Dürüst CV doğruluğu    : {k_katli_dogruluk(X, y):.2f}")
```

**Çıktı:**

```text
Sızıntılı CV doğruluğu : 0.92
Dürüst CV doğruluğu    : 0.53
```

### Neden böyle oldu?

2000 rastgele özelliğin içinde, **tesadüfen** etiketle korele görünen birkaç düzine özellik mutlaka vardır. Seçimi tüm veriyle yaptığımızda, bu tesadüfi korelasyon **test örneklerini de kapsıyor** — yani test etiketleri seçim adımına sızmış oluyor. Çapraz doğrulama artık modeli değil, kendi sızıntımızı ölçüyor.

Doğru yolda ise seçim test katını hiç görmüyor; tesadüfi korelasyonlar yeni veriye genellenmiyor ve doğruluk olması gereken yere, ≈ %50'ye iniyor.

> 💡 Bu örnek, *The Elements of Statistical Learning* kitabının "çapraz doğrulamayı yanlış yapmanın yolu" bölümündeki klasik gösterimin sadeleştirilmiş hâlidir.

## 4. Altın Kural

**Veriden öğrenen her adım** (ortalama/standart sapma hesaplamak, eksik değer doldurmak, kategori kodlamak, özellik seçmek, PCA, SMOTE ile çoğaltmak…) bir *model parçası* gibi düşünülmelidir:

1. Önce veriyi böl (train / validation / test).
2. Adımı **yalnızca eğitim verisiyle** `fit` et.
3. Aynı dönüşümü validation ve test verisine sadece `transform` olarak uygula.
4. Çapraz doğrulamada bu işlemi **her katta yeniden** yap.

scikit-learn kullanıyorsanız bunun en güvenli yolu tüm adımları bir `Pipeline` içine koyup `cross_val_score` / `GridSearchCV`'ye **pipeline'ı** vermektir:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression())
skorlar = cross_val_score(model, X, y, cv=5)  # ölçekleme ve seçim her katta yeniden fit edilir
```

## 5. Sık Yapılan Hatalar (Kontrol Listesi)

- [ ] `StandardScaler().fit(X)` → **sonra** `train_test_split` (sıra ters olmalı).
- [ ] Eksik değerleri tüm verinin ortalamasıyla doldurmak.
- [ ] Dengesiz veride SMOTE / oversampling'i bölmeden önce yapmak — çoğaltılan örneklerin "kardeşleri" test setine düşer.
- [ ] Zaman serisinde rastgele karıştırarak bölmek: model geleceği görerek geçmişi tahmin eder. Zamana göre böl (`TimeSeriesSplit`).
- [ ] Aynı müşteriye/hastaya ait birden fazla satırın hem eğitimde hem testte bulunması. Gruba göre böl (`GroupKFold`).
- [ ] Hiperparametreleri test setine bakarak ayarlamak. Test seti **en sonda, bir kez** kullanılır; ayarlama için validation veya çapraz doğrulama kullanılır.
- [ ] Hedefin sonradan oluşan bir türevini özellik olarak bırakmak (ör. kredi temerrüdü tahmininde "tahsilat_birimine_devredildi").

**Uyarı işareti:** Beklenmedik derecede yüksek bir skor (%99 doğruluk, 0.99 AUC) ya da tek bir özelliğin önem sıralamasında açık ara birinci çıkması. Böyle bir sonuç gördüğünüzde ilk şüpheli sızıntıdır.

## 6. Alıştırma

Bir özelliğin eğitim değerleri `[12, 15, 11, 14, 13]`, test değerleri `[30, 16]`.

1. Z-skor standardizasyonunu **doğru** şekilde uygulayın (popülasyon standart sapması ile).
2. Test setindeki 30 değeri ölçeklenmiş hâlde neden çok büyük çıkıyor? Bu bir sorun mu?

<details>
<summary><b>Çözüm</b></summary>

1. Ortalama ve standart sapma **yalnızca eğitim verisinden** hesaplanır:

```python
import statistics

egitim = [12.0, 15.0, 11.0, 14.0, 13.0]
test = [30.0, 16.0]

# Ölçekleyiciyi SADECE eğitim verisiyle "fit" et
ort = statistics.fmean(egitim)
std = statistics.pstdev(egitim)

def olcekle(degerler):
    return [round((v - ort) / std, 2) for v in degerler]

print("ortalama:", ort, " std:", round(std, 3))
print("eğitim  :", olcekle(egitim))
print("test    :", olcekle(test))
```

```text
ortalama: 13.0  std: 1.414
eğitim  : [-0.71, 1.41, -1.41, 0.71, 0.0]
test    : [12.02, 2.12]
```

2. 30 değeri, eğitim dağılımına göre ortalamadan ≈ 12 standart sapma uzakta — yani gerçekten **aykırı** bir gözlem. Bu bir hata değil, doğru davranıştır: Model canlıda da yeni gelen veriyi eğitimde gördüğü dağılıma göre ölçekleyecektir. Eğer ortalama/std'yi test dahil tüm veriyle hesaplasaydık (ortalama ≈ 15.86), bu aykırılık yumuşatılır ve model değerlendirmesi gerçek hayattan farklı bir durumu ölçerdi. Aykırı değerler ayrıca ele alınmalı (izleme, kırpma, robust ölçekleme), ama test verisini ölçekleyiciye "göstererek" değil.

</details>

## 7. Özet

- Sızıntı = tahmin anında olmayacak bilginin eğitime karışması.
- Belirtisi: Doğrulamada harika, canlıda kötü performans.
- Çözüm: **Önce böl, sonra fit et**; tüm ön işleme adımlarını pipeline içinde, çapraz doğrulamanın her katında yeniden çalıştır.
- Zaman ve grup yapısı varsa bölme stratejisini ona göre seç.
