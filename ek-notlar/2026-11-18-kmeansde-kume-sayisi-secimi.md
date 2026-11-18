# K-Means'te Küme Sayısını (k) Seçmek: Dirsek Yöntemi ve Silhouette Skoru

> **Bağlantılı olduğu dersler:** Ders 3 – K-Means ve Parsing · Ders 5 – Müşteri Segmentasyonu
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Problem: K-Means "kaç küme?" sorusunu kendisi cevaplamaz

K-Means algoritmasına küme sayısı **k** dışarıdan verilir. Algoritma her k için bir sonuç üretir — k = 2 de, k = 10 da "çalışır". Peki müşterilerimizi 3 segmente mi, 4 segmente mi ayırmalıyız? Bu notta bu soruyu cevaplamak için en çok kullanılan iki yöntemi sıfırdan uyguluyoruz:

1. **Dirsek (Elbow) yöntemi** – inertia'nın k'ya göre değişimi
2. **Silhouette skoru** – kümelerin ne kadar "sıkı" ve birbirinden ne kadar "ayrık" olduğu

## 2. Kısa Hatırlatma: K-Means Adımları

1. Rastgele **k** merkez seç.
2. **Atama:** Her noktayı en yakın merkeze ata.
3. **Güncelleme:** Her merkezi, kendisine atanan noktaların ortalamasına taşı.
4. Merkezler değişmeyene kadar 2–3'ü tekrarla.

Sonuç rastgele başlangıca bağlı olduğundan algoritma birkaç kez farklı başlangıçlarla çalıştırılır ve en iyi sonuç alınır (scikit-learn'deki `n_init` parametresi).

## 3. İki Ölçüt

### Inertia (WCSS – küme içi kareler toplamı)

$$\text{Inertia} = \sum_{i} \lVert x_i - \mu_{c(i)} \rVert^2$$

Her noktanın kendi küme merkezine uzaklığının karelerinin toplamıdır. **k arttıkça inertia her zaman azalır** (k = n olduğunda sıfır olur). Bu yüzden "en düşük inertia" diye seçim yapılamaz; bunun yerine azalmanın **birden yavaşladığı** nokta — grafikteki "dirsek" — aranır.

### Silhouette Skoru

Her nokta *i* için:

- **a(i):** Kendi kümesindeki diğer noktalara ortalama uzaklık (*ne kadar sıkı?*)
- **b(i):** En yakın **diğer** kümenin noktalarına ortalama uzaklık (*ne kadar ayrık?*)

$$s(i) = \frac{b(i) - a(i)}{\max\big(a(i),\, b(i)\big)}$$

| s(i) değeri | Anlamı |
|---|---|
| ≈ +1 | Nokta kendi kümesine çok iyi oturuyor |
| ≈ 0 | İki kümenin sınırında |
| < 0 | Muhtemelen yanlış kümede |

Tüm noktaların ortalaması alınır; **en yüksek ortalama silhouette** veren k tercih edilir. Inertia'dan farklı olarak k ile monoton değişmez, bu yüzden doğrudan karşılaştırma yapılabilir.

## 4. Uygulama: Sıfırdan Dirsek ve Silhouette

Aşağıda 4 gerçek gruptan oluşan sentetik bir "müşteri" verisi (yıllık harcama, ziyaret sıklığı) üretiliyor ve k = 2…7 için iki ölçüt hesaplanıyor.

```python
import math
import random

random.seed(7)

# Sentetik müşteri verisi: (yıllık harcama, ziyaret sıklığı) — 4 gerçek grup
merkezler = [(2, 2), (8, 3), (5, 8), (9, 9)]
X = [(random.gauss(cx, 0.8), random.gauss(cy, 0.8))
     for cx, cy in merkezler for _ in range(40)]


def uzaklik(a, b):
    return math.dist(a, b)


def kmeans(X, k, tekrar=10, iterasyon=100):
    """Birden fazla rastgele başlangıçla K-Means; en düşük inertia'lı sonucu döndürür."""
    en_iyi = None
    for _ in range(tekrar):
        merkez = random.sample(X, k)
        for _ in range(iterasyon):
            # 1) Atama adımı: her nokta en yakın merkeze
            etiket = [min(range(k), key=lambda j: uzaklik(x, merkez[j])) for x in X]
            # 2) Güncelleme adımı: merkez = kümedeki noktaların ortalaması
            yeni = []
            for j in range(k):
                uyeler = [x for x, e in zip(X, etiket) if e == j]
                if uyeler:
                    yeni.append(tuple(sum(v) / len(uyeler) for v in zip(*uyeler)))
                else:  # boş küme: rastgele bir noktayla yeniden başlat
                    yeni.append(random.choice(X))
            if yeni == merkez:
                break
            merkez = yeni
        inertia = sum(uzaklik(x, merkez[e]) ** 2 for x, e in zip(X, etiket))
        if en_iyi is None or inertia < en_iyi[0]:
            en_iyi = (inertia, etiket, merkez)
    return en_iyi


def silhouette(X, etiket):
    """Ortalama silhouette skoru: s = (b - a) / max(a, b)."""
    kumeler = set(etiket)
    skorlar = []
    for i, x in enumerate(X):
        ayni = [uzaklik(x, y) for j, y in enumerate(X) if etiket[j] == etiket[i] and j != i]
        if not ayni:          # tek elemanlı küme: tanım gereği s = 0
            skorlar.append(0.0)
            continue
        a = sum(ayni) / len(ayni)
        b = min(
            sum(uzaklik(x, y) for j, y in enumerate(X) if etiket[j] == c) /
            sum(1 for e in etiket if e == c)
            for c in kumeler if c != etiket[i]
        )
        skorlar.append((b - a) / max(a, b))
    return sum(skorlar) / len(skorlar)


print(" k   inertia   düşüş(%)  silhouette")
onceki = None
for k in range(2, 8):
    inertia, etiket, _ = kmeans(X, k)
    dusus = "" if onceki is None else f"{100 * (onceki - inertia) / onceki:6.1f}"
    print(f"{k:2d}  {inertia:8.1f}  {dusus:>8}  {silhouette(X, etiket):9.3f}")
    onceki = inertia
```

### Çıktı

```text
 k   inertia   düşüş(%)  silhouette
 2    1304.5                0.524
 3     505.9      61.2      0.647
 4     210.2      58.5      0.684
 5     183.5      12.7      0.602
 6     162.9      11.3      0.495
 7     140.8      13.6      0.482
```

## 5. Sonuçların Yorumu

1. **Dirsek:** k = 2 → 3 geçişinde inertia %61, 3 → 4 geçişinde %58,5 düşüyor; 4'ten sonra düşüş birden **%11–14** bandına iniyor. Dirsek açıkça **k = 4**'te.
2. **Silhouette:** En yüksek değer **k = 4**'te (0.684). k = 5'te bir gerçek grup yapay olarak ikiye bölündüğü için skor düşüyor (0.602).
3. İki yöntem aynı cevabı veriyor ve bu cevap veriyi ürettiğimiz 4 gerçek merkezle örtüşüyor. Gerçek verilerde dirsek bu kadar net olmayabilir; o zaman silhouette (ve iş bilgisi) karar vermede daha güvenilirdir.

## 6. Sık Yapılan Hatalar

- ❌ **En düşük inertia'yı seçmek.** Inertia k ile her zaman azalır; bu ölçütle her zaman en büyük k "kazanır".
- ❌ **Özellikleri ölçeklememek.** K-Means Öklid uzaklığı kullanır. "Yıllık gelir (TL)" ile "yaş" aynı tabloda ölçeklenmeden kullanılırsa küme yapısını neredeyse tamamen gelir belirler. Önce `StandardScaler` gibi bir ölçekleme uygulanmalıdır.
- ❌ **Tek bir rastgele başlangıçla çalıştırmak.** Kötü bir başlangıç yerel minimuma takılabilir; birden çok başlangıç (`n_init`) ve `k-means++` başlatma kullanılmalıdır.
- ❌ **Kümelerin küresel olmadığı veride K-Means'e güvenmek.** Hilal, halka gibi şekillerde veya çok farklı yoğunluktaki kümelerde K-Means (ve silhouette) yanıltıcı olur; DBSCAN gibi yöntemler denenmelidir.
- ❌ **İstatistiği iş bağlamının önüne koymak.** Silhouette k = 4 dese bile pazarlama ekibi yalnızca 3 kampanya yürütebiliyorsa, k = 3'ün skoru kabul edilebilir düzeydeyse o tercih edilebilir.

## 7. scikit-learn ile Kısa Yol

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X_olcekli = StandardScaler().fit_transform(X)

for k in range(2, 8):
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_olcekli)
    print(k, round(model.inertia_, 1), round(silhouette_score(X_olcekli, model.labels_), 3))
```

Not: Silhouette hesabı tüm nokta çiftleri arasındaki uzaklığı kullandığı için O(n²) maliyetlidir. Çok büyük veride `silhouette_score(..., sample_size=5000)` ile örneklem üzerinden hesaplanabilir.

## 8. Mini Alıştırma

**Soru:** A noktası 1. kümededir. A'nın 1. kümedeki diğer iki noktaya uzaklığı **1** ve **3**'tür. 2. kümedeki noktalara uzaklıkları **6** ve **8**, 3. kümedeki noktalara uzaklıkları **9** ve **11**'dir. A'nın silhouette değeri nedir? A doğru kümede mi?

<details>
<summary>Çözüm</summary>

- a(A) = (1 + 3) / 2 = **2**
- 2. kümeye ortalama uzaklık = (6 + 8) / 2 = 7; 3. kümeye = (9 + 11) / 2 = 10 → en yakın diğer küme 2. küme, **b(A) = 7**
- s(A) = (7 − 2) / max(2, 7) = 5 / 7 ≈ **0.714**

Değer pozitif ve 1'e yakın olduğundan A kendi kümesine iyi oturuyor; yani doğru kümede.
</details>

---
*Özet:* k'yı seçerken inertia'nın "dirseğine" ve ortalama silhouette skoruna birlikte bak, veriyi önceden ölçekle ve son kararı iş bağlamıyla doğrula.
