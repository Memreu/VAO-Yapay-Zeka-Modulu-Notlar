# K-Katlı Çapraz Doğrulama ile Model Seçimi: Polinom Derecesini Veriye Sordurmak

> **Bağlantılı olduğu dersler:** Ders 2 – Denetimli Öğrenme (Curve Fitting) · Ders 4–5 – Aşırı öğrenme, düzenlileştirme ve model değerlendirme
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Problem: Eğitim Hatası Yalan Söyler

Ders 2'deki *curve fitting* örneğinde polinomun derecesini artırdıkça eğri noktalara daha iyi oturur. Eğitim hatası (eğitim MSE) **her zaman** azalır, çünkü derecesi yüksek polinom, düşük dereceli polinomu özel durum olarak zaten içerir. Bu yüzden "eğitim hatası en düşük modeli seç" kuralı bizi her seferinde en karmaşık modele, yani **aşırı öğrenmeye** (*overfitting*) götürür.

Asıl merak ettiğimiz şey **genelleme hatasıdır**: Model, hiç görmediği yeni verilerde ne kadar hata yapar? Bunu ölçmenin iki yolu var:

| Yöntem | Fikir | Zayıf yanı |
|---|---|---|
| Tek doğrulama kümesi (*hold-out*) | Veriyi bir kez %80 / %20 böl | Az veride sonuç, bölmenin şansına çok bağlıdır; verinin %20'si eğitime hiç girmez |
| **K-katlı çapraz doğrulama** (*k-fold CV*) | Veriyi $k$ parçaya böl, her parçayı sırayla doğrulama kümesi yap | $k$ kat daha fazla eğitim gerekir |

## 2. Algoritma Adım Adım

1. Veriyi **karıştır** (sıralı veride bu adım atlanırsa katlar birbirinden çok farklı olur).
2. $k$ eşit parçaya (kat, *fold*) böl. Tipik değerler: $k = 5$ veya $k = 10$.
3. Her $j = 1, \dots, k$ için:
   - $j$. katı **doğrulama**, geri kalan $k-1$ katı **eğitim** kümesi yap,
   - modeli yalnızca eğitim kısmında fit et,
   - doğrulama katındaki hatayı $E_j$ olarak kaydet.
4. CV skoru: $\displaystyle \text{CV}_k = \frac{1}{k}\sum_{j=1}^{k} E_j$, yanında katlar arası standart sapma.
5. Bunu her aday model (burada her polinom derecesi) için tekrarla; **CV skoru en düşük** olanı seç.
6. Seçilen ayarla modeli **tüm veri** üzerinde yeniden eğit.

Her gözlem tam olarak bir kez doğrulamada, $k-1$ kez eğitimde kullanılır. $k = N$ seçilirse buna **birini dışarıda bırak** (*leave-one-out*, LOOCV) denir.

## 3. Çalışan Örnek: Hangi Derece?

Veri, $y = \sin(\pi x) + \text{gürültü}$ kuralından gelen sadece **30 nokta**. 0'dan 8'e kadar her polinom derecesi için eğitim MSE'sini, 5-katlı CV MSE'sini ve karşılaştırma için 2000 noktalık, modelin hiç görmediği bir test kümesindeki MSE'yi yazdırıyoruz. (Gerçek hayatta bu büyük test kümesi elimizde olmaz; burada sadece CV'nin doğru tahmin yapıp yapmadığını kontrol etmek için var.)

```python
import math
import random

random.seed(0)

# Gerçek ilişki: y = sin(πx) + gürültü,  x ∈ [-1, 1]
N = 30
X = [random.uniform(-1, 1) for _ in range(N)]
Y = [math.sin(math.pi * x) + random.gauss(0, 0.3) for x in X]


def coz(A, c):
    """Gauss eliminasyonu (kısmi pivotlama) ile A·β = c sistemini çözer."""
    n = len(A)
    M = [satir[:] + [ci] for satir, ci in zip(A, c)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    beta = [0.0] * n
    for i in range(n - 1, -1, -1):
        beta[i] = (M[i][n] - sum(M[i][j] * beta[j] for j in range(i + 1, n))) / M[i][i]
    return beta


def polinom_fit(xs, ys, derece):
    """En küçük kareler: (XᵀX) β = Xᵀy normal denklemleri."""
    d = derece + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(d)] for i in range(d)]
    c = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(d)]
    return coz(A, c)


def tahmin(beta, x):
    return sum(b * x ** i for i, b in enumerate(beta))


def mse(beta, xs, ys):
    return sum((tahmin(beta, x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def k_katli_cv(xs, ys, derece, k=5, tohum=1):
    """Veriyi k parçaya böl; her parça bir kez doğrulama, kalanı eğitim olsun."""
    idx = list(range(len(xs)))
    random.Random(tohum).shuffle(idx)          # karıştır, sonra böl
    katlar = [idx[i::k] for i in range(k)]
    hatalar = []
    for dogrulama in katlar:
        egitim = [i for i in idx if i not in dogrulama]
        beta = polinom_fit([xs[i] for i in egitim], [ys[i] for i in egitim], derece)
        hatalar.append(mse(beta, [xs[i] for i in dogrulama], [ys[i] for i in dogrulama]))
    ort = sum(hatalar) / k
    std = math.sqrt(sum((h - ort) ** 2 for h in hatalar) / (k - 1))
    return ort, std


# Modelin hiç görmediği, aynı kuraldan gelen büyük bir test kümesi
rt = random.Random(99)
XT = [rt.uniform(-1, 1) for _ in range(2000)]
YT = [math.sin(math.pi * x) + rt.gauss(0, 0.3) for x in XT]

print(f"{'derece':>6} | {'eğitim MSE':>10} | {'5-katlı CV MSE':>20} | {'test MSE':>8}")
print("-" * 56)
sonuc = {}
for d in range(0, 9):
    beta = polinom_fit(X, Y, d)
    egt = mse(beta, X, Y)
    cv, s = k_katli_cv(X, Y, d)
    tst = mse(beta, XT, YT)
    sonuc[d] = (cv, s)
    print(f"{d:>6} | {egt:>10.4f} | {cv:>9.3f} ± {s:<8.3f} | {tst:>8.4f}")

en_iyi = min(sonuc, key=lambda d: sonuc[d][0])
print(f"\nCV'ye göre en iyi derece: {en_iyi}")

# Tek standart hata kuralı: en iyi CV'nin 1 SE içindeki EN BASİT model
cv_min, s_min = sonuc[en_iyi]
se = s_min / math.sqrt(5)
basit = min(d for d in sonuc if sonuc[d][0] <= cv_min + se)
print(f"1-SE kuralına göre seçilen derece: {basit}")

# Karıştırma tohumu değişince CV seçimi değişiyor mu?
for t in range(1, 6):
    secim = min(range(0, 9), key=lambda d: k_katli_cv(X, Y, d, tohum=t)[0])
    print(f"tohum={t} → seçilen derece {secim}")
```

**Çıktı:**

```
derece | eğitim MSE |       5-katlı CV MSE | test MSE
--------------------------------------------------------
     0 |     0.4875 |     0.494 ± 0.191    |   0.6443
     1 |     0.1903 |     0.234 ± 0.133    |   0.2867
     2 |     0.1739 |     0.334 ± 0.201    |   0.3625
     3 |     0.0939 |     0.120 ± 0.060    |   0.1125
     4 |     0.0930 |     0.285 ± 0.321    |   0.1086
     5 |     0.0898 |     2.312 ± 4.641    |   0.1421
     6 |     0.0838 |     2.705 ± 5.500    |   1.0470
     7 |     0.0825 |    45.551 ± 98.279   |   4.4215
     8 |     0.0808 |   492.224 ± 1093.939 |  28.7264

CV'ye göre en iyi derece: 3
1-SE kuralına göre seçilen derece: 3
tohum=1 → seçilen derece 3
tohum=2 → seçilen derece 3
tohum=3 → seçilen derece 3
tohum=4 → seçilen derece 3
tohum=5 → seçilen derece 3
```

## 4. Çıktıyı Okumak

- **Eğitim MSE'si derece arttıkça sürekli düşüyor** (0.49 → 0.08). Buna bakarak 8. dereceyi seçerdik; oysa o modelin test hatası 28.7, yani felaket.
- **CV MSE'si U şeklinde:** Düşük derecelerde model veriyi yakalayamıyor (*underfitting*, yüksek yanlılık). 3. derecede en düşük değere iniyor. Daha sonra yeniden yükseliyor (*overfitting*, yüksek varyans). CV, test kümesine hiç bakmadan doğru bölgeyi buldu.
- **2. derece neden 1. dereceden kötü?** $\sin(\pi x)$ tek bir fonksiyondur. $x^2$ terimi simetrik olduğu için ona yardım etmez, sadece gürültüyü öğrenir. 3. derecedeki $x^3$ terimi ise sinüsün kıvrımını yakalar.
- **Yüksek derecede CV neden bu kadar patlıyor?** Rastgele bölünen bir katta $x$ aralığının uç noktaları doğrulamaya düşerse, model o bölgede **dış değerleme** (*extrapolation*) yapmak zorunda kalır. Yüksek dereceli polinomlar ise aralığın dışında çok hızlı büyür. Bu yüzden ortalamanın yanında **standart sapmaya** da bakmak gerekir. ± değeri ortalamadan büyükse modelin kararsız olduğu anlaşılır.
- **4. derecenin test hatası (0.109), 3. dereceninkinden (0.113) biraz daha düşük.** Aradaki fark gürültü düzeyinde. CV ise 4. derecede katlar arası belirsizliği çok daha yüksek gösteriyor. Fark bu kadar küçükken **daha basit modeli** tercih etmek mantıklıdır.
- **Tohum (karıştırma) değişince seçim değişmiyor:** Beş farklı bölmenin hepsi 3. dereceyi seçti, yani sonuç tek bir şanslı bölmeye bağlı değil.

### Tek Standart Hata (1-SE) Kuralı

Pratikte sık kullanılan bir sezgisel kural: En iyi CV skorunu bul. Sonra skoru $\text{CV}_{\min} + \text{SE}$ sınırının altında kalan modeller arasından **en basit olanı** seç. Burada $\text{SE} = s / \sqrt{k}$, en iyi modelin katlar arası standart hatasıdır. Bu kural, CV'nin kendi gürültüsünden dolayı gereğinden karmaşık bir model seçmeyi önler. Örneğimizde iki kural da aynı sonucu (3. derece) verdi.

## 5. $k$ Nasıl Seçilir?

| $k$ | Eğitim kümesi boyutu | Yanlılık | Varyans | Maliyet |
|---|---|---|---|---|
| 2–3 | Verinin %50–67'si | Yüksek (model az veriyle eğitilir, hatayı abartır) | Düşük | Ucuz |
| **5–10** | **%80–90** | **Makul** | **Makul** | **Orta (pratikte varsayılan)** |
| $N$ (LOOCV) | $N-1$ | Çok düşük | Yüksek (eğitim kümeleri birbirine çok benzer) | $N$ kez eğitim |

## 6. Sık Yapılan Hatalar

- ❌ **Ön işlemeyi CV'den önce tüm veriye uygulamak.** Ölçekleme, eksik değer doldurma veya özellik seçimi her katta **yalnızca eğitim kısmında** fit edilmelidir. Aksi halde doğrulama katının bilgisi eğitime sızar (bkz. [Veri Sızıntısı notu](2026-11-18-veri-sizintisi-data-leakage.md)). scikit-learn'de bunun yolu `Pipeline` + `cross_val_score` kullanmaktır.
- ❌ **Model seçtikten sonra aynı CV skorunu "nihai başarı" diye raporlamak.** Çok sayıda aday arasından en iyisini seçtiğimiz için bu skor biraz iyimserdir. Dürüst bir tahmin için ya en başta ayrılmış, hiç dokunulmamış bir **test kümesi** ya da **iç içe CV** (*nested CV*) kullanılmalıdır.
- ❌ **Zaman serisinde karıştırarak bölmek.** Gelecekteki veriyle geçmişi tahmin etmiş olursunuz. Zaman serilerinde katlar zaman sırasına göre ilerlemelidir (`TimeSeriesSplit`).
- ❌ **Dengesiz sınıflarda düz K-fold kullanmak.** Bazı katlarda azınlık sınıfı hiç bulunmayabilir. Bu durumda her katta sınıf oranlarını koruyan `StratifiedKFold` kullanılmalıdır (bkz. [Dengesiz Veri notu](2026-11-17-dengesiz-veride-dogru-metrik-secimi.md)).
- ❌ **Aynı kişiye veya nesneye ait satırları farklı katlara dağıtmak.** Örneğin aynı müşterinin birden çok kaydı varsa model kişiyi ezberler. Bu durumda `GroupKFold` kullanılmalıdır.
- ❌ **Sadece ortalamaya bakmak.** Katlar arası standart sapma, iki model arasındaki farkın anlamlı olup olmadığını gösterir.

## 7. Mini Alıştırma

**Soru:** 1000 satırlık bir veri setinde 4 farklı model ve her model için 6 farklı hiperparametre değeri deneyeceksiniz.

1. 5-katlı CV ile toplam kaç kez model eğitilir (son yeniden eğitim hariç)?
2. LOOCV ile kaç kez?
3. Bir arkadaşınız CV skorları birbirine çok yakın çıkan iki modelden ($0.120 \pm 0.060$ ve $0.118 \pm 0.150$) ikincisini seçmek istiyor. Ne önerirsiniz?

<details>
<summary>Çözüm</summary>

1. Aday sayısı $4 \times 6 = 24$. Her aday için 5 eğitim yapılır: $24 \times 5 = 120$ eğitim.
2. LOOCV'de $k = N = 1000$ olur: $24 \times 1000 = 24\,000$ eğitim. Bu, $200$ kat daha pahalıdır ve büyük verilerde genellikle gereksizdir.
3. İki skor arasındaki fark ($0.002$), katlar arası standart sapmanın yanında önemsiz kalıyor. İkinci model ayrıca çok daha kararsız ($\pm 0.150$). 1-SE kuralının mantığına göre **daha basit veya daha kararlı olan** birinci model tercih edilmelidir. Emin olmak için CV'yi farklı tohumlarla tekrarlamak da faydalıdır (*repeated k-fold*).
</details>

---
*Özet:* Eğitim hatası model karmaşıklaştıkça hep düşer, bu yüzden model seçiminde kullanılamaz. K-katlı çapraz doğrulama her gözlemi bir kez doğrulamada kullanarak genelleme hatasını tahmin eder. CV skoru en düşük olan modeli seç, standart sapmaya da bak. Skorlar birbirine yakınsa basit modeli tercih et ve ön işlemeyi her zaman katların içinde yap.
