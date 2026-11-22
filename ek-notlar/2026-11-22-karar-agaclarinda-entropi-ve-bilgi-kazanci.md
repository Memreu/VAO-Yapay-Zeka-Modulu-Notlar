# Karar Ağaçlarında Entropi, Gini ve Bilgi Kazancı: Ağaç Hangi Soruyu Önce Sorar?

> **Bağlantılı olduğu dersler:** Ders 4 – Karar ağacı (*Decision Tree*) ile meyve sınıflandırma örneği · Ders 2 – Denetimli öğrenme
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Problem: Hangi Özellik Köke Gelmeli?

Ders 4'te bilgisayara meyveleri boyut, renk ve şekil gibi özelliklerine bakarak tanıtan bir karar ağacı gördük. Her iç düğüm bir soru sorar ("Boyutu ne?"), her yaprak bir sınıf söyler ("elma"). Asıl soru şu: **Ağaç önce hangi soruyu soracağına nasıl karar veriyor?**

İyi bir soru, veriyi sınıfları birbirinden olabildiğince **ayrışmış** (saf) gruplara böler. "Saflığı" sayıyla ölçebilirsek, her düğümde safsızlığı **en çok azaltan** soruyu seçebiliriz. Ağaç algoritmalarının (ID3, C4.5, CART) hepsi bu açgözlü (*greedy*) fikre dayanır.

## 2. Safsızlık Ölçüleri

Bir düğümde $C$ sınıf olsun ve $p_i$, $i$. sınıfın o düğümdeki oranı olsun.

| Ölçü | Formül | Saf düğüm | En karışık durum (2 sınıf, 50/50) | Kullanan |
|---|---|---|---|---|
| **Entropi** | $H(S) = -\sum_i p_i \log_2 p_i$ | $0$ | $1$ bit | ID3, C4.5 |
| **Gini safsızlığı** | $G(S) = 1 - \sum_i p_i^2$ | $0$ | $0.5$ | CART (scikit-learn varsayılanı) |

**Sezgi:**
- **Entropi**, "Bu düğümden rastgele bir örnek çeksem, sınıfını öğrenmek için ortalama kaç evet/hayır sorusu gerekir?" sorusunun cevabıdır. 50/50 bir para atışı 1 bit belirsizlik taşır. %90/%10 dağılımda belirsizlik $0.469$ bite düşer.
- **Gini**, "Rastgele bir örnek çekip ona yine düğümün dağılımına göre rastgele etiket versem, yanılma olasılığım ne?" sorusunun cevabıdır. %90/%10 için $1 - 0.81 - 0.01 = 0.18$.

İki ölçü pratikte çoğu zaman aynı bölmeyi seçer. Gini logaritma içermediği için biraz daha hızlı hesaplanır.

## 3. Bilgi Kazancı (*Information Gain*)

$A$ özelliğine göre böldüğümüzde $S$ kümesi $S_v$ alt kümelerine ayrılır. Çocukların safsızlığı, **boyutlarıyla ağırlıklandırılarak** ortalanır:

$$
\text{IG}(S, A) = H(S) - \sum_{v \in \text{değerler}(A)} \frac{|S_v|}{|S|}\, H(S_v)
$$

Her düğümde IG'si en büyük özellik seçilir. Çocuklar için aynı işlem tekrarlanır. Bir düğüm **saf** olunca ya da sorulacak özellik kalmayınca durulur.

### Elle Bir Hesap

Aşağıdaki 12 meyvelik veride sınıf sayıları şöyle: karpuz 2, kavun 2, elma 3, muz 2, kiraz 2, üzüm 1.

$$
H(S) = -\left[5 \cdot \tfrac{2}{12}\log_2\tfrac{2}{12} + \tfrac{3}{12}\log_2\tfrac{3}{12} + \tfrac{1}{12}\log_2\tfrac{1}{12}\right] \approx 2.522 \text{ bit}
$$

`boyut` sorusu veriyi üçe böler:

| boyut | Örnek | İçerik | Entropi |
|---|---|---|---|
| büyük | 4 | 2 karpuz, 2 kavun | $1.000$ |
| orta | 5 | 3 elma, 2 muz | $0.971$ |
| küçük | 3 | 2 kiraz, 1 üzüm | $0.918$ |

Ağırlıklı çocuk entropisi: $\tfrac{4}{12}(1.000) + \tfrac{5}{12}(0.971) + \tfrac{3}{12}(0.918) \approx 0.968$.

Kazanç: $\text{IG} = 2.522 - 0.968 = 1.555$ bit.

## 4. Çalışan Örnek: Sıfırdan Küçük Bir ID3

```python
import math
from collections import Counter

# Ders 4'teki meyve örneğinin küçük bir versiyonu: (boyut, renk, şekil) -> meyve
VERI = [
    ("büyük", "yeşil",   "yuvarlak", "karpuz"),
    ("büyük", "yeşil",   "yuvarlak", "karpuz"),
    ("büyük", "sarı",    "yuvarlak", "kavun"),
    ("büyük", "sarı",    "uzun",     "kavun"),
    ("orta",  "kırmızı", "yuvarlak", "elma"),
    ("orta",  "yeşil",   "yuvarlak", "elma"),
    ("orta",  "kırmızı", "yuvarlak", "elma"),
    ("orta",  "sarı",    "uzun",     "muz"),
    ("orta",  "sarı",    "uzun",     "muz"),
    ("küçük", "kırmızı", "yuvarlak", "kiraz"),
    ("küçük", "kırmızı", "yuvarlak", "kiraz"),
    ("küçük", "yeşil",   "yuvarlak", "üzüm"),
]
OZELLIKLER = ["boyut", "renk", "şekil"]


def entropi(etiketler):
    """H(S) = -Σ p_i · log2(p_i)  (bit cinsinden)"""
    n = len(etiketler)
    return -sum((c / n) * math.log2(c / n) for c in Counter(etiketler).values())


def gini(etiketler):
    """G(S) = 1 - Σ p_i²"""
    n = len(etiketler)
    return 1 - sum((c / n) ** 2 for c in Counter(etiketler).values())


def bolme_kazanci(satirlar, j, olcu=entropi):
    """j. özelliğe göre bölünce safsızlıktaki azalma (ağırlıklı)."""
    ebeveyn = olcu([s[-1] for s in satirlar])
    gruplar = {}
    for s in satirlar:
        gruplar.setdefault(s[j], []).append(s[-1])
    cocuk = sum(len(g) / len(satirlar) * olcu(g) for g in gruplar.values())
    return ebeveyn - cocuk


def agac_kur(satirlar, ozellikler, derinlik=0):
    etiketler = [s[-1] for s in satirlar]
    girinti = "    " * derinlik
    if len(set(etiketler)) == 1:                       # saf düğüm -> yaprak
        print(f"{girinti}→ {etiketler[0]}  ({len(etiketler)} örnek)")
        return
    if not ozellikler:                                 # soracak özellik kalmadı
        print(f"{girinti}→ {Counter(etiketler).most_common(1)[0][0]}  (çoğunluk)")
        return
    kazanclar = {o: bolme_kazanci(satirlar, OZELLIKLER.index(o)) for o in ozellikler}
    en_iyi = max(kazanclar, key=kazanclar.get)
    j = OZELLIKLER.index(en_iyi)
    ozet = ", ".join(f"{o}={k:.3f}" for o, k in kazanclar.items())
    print(f"{girinti}[{en_iyi}?]  H={entropi(etiketler):.3f}  kazançlar: {ozet}")
    for deger in sorted({s[j] for s in satirlar}):
        alt = [s for s in satirlar if s[j] == deger]
        print(f"{girinti}  {en_iyi} = {deger}:")
        agac_kur(alt, [o for o in ozellikler if o != en_iyi], derinlik + 1)


etiketler = [s[-1] for s in VERI]
print(f"Kök düğüm: H = {entropi(etiketler):.3f} bit, Gini = {gini(etiketler):.3f}\n")

print("Kökte her özelliğin kazancı:")
for j, o in enumerate(OZELLIKLER):
    print(f"  {o:6s}  bilgi kazancı = {bolme_kazanci(VERI, j):.3f}   "
          f"Gini azalması = {bolme_kazanci(VERI, j, gini):.3f}")

print("\nKurulan ağaç (ID3):")
agac_kur(VERI, OZELLIKLER)
```

**Çıktı:**

```text
Kök düğüm: H = 2.522 bit, Gini = 0.819

Kökte her özelliğin kazancı:
  boyut   bilgi kazancı = 1.555   Gini azalması = 0.342
  renk    bilgi kazancı = 1.355   Gini azalması = 0.278
  şekil   bilgi kazancı = 0.645   Gini azalması = 0.134

Kurulan ağaç (ID3):
[boyut?]  H=2.522  kazançlar: boyut=1.555, renk=1.355, şekil=0.645
  boyut = büyük:
    [renk?]  H=1.000  kazançlar: renk=1.000, şekil=0.311
      renk = sarı:
        → kavun  (2 örnek)
      renk = yeşil:
        → karpuz  (2 örnek)
  boyut = küçük:
    [renk?]  H=0.918  kazançlar: renk=0.918, şekil=0.000
      renk = kırmızı:
        → kiraz  (2 örnek)
      renk = yeşil:
        → üzüm  (1 örnek)
  boyut = orta:
    [renk?]  H=0.971  kazançlar: renk=0.971, şekil=0.971
      renk = kırmızı:
        → elma  (2 örnek)
      renk = sarı:
        → muz  (2 örnek)
      renk = yeşil:
        → elma  (1 örnek)
```

## 5. Çıktıyı Okumak

- **Kökte `boyut` seçildi** (IG = 1.555). Elle yaptığımız hesapla birebir aynı. Gini azalmasına göre sıralama da aynı (boyut > renk > şekil), yani iki ölçü burada aynı ağacı kurar.
- `şekil` en zayıf soru. Neredeyse her meyve yuvarlak olduğu için grupları ayırmıyor.
- `boyut = küçük` dalında `şekil`'in kazancı **tam 0**. Kiraz da üzüm de yuvarlak olduğundan bu soru hiçbir şey öğretmiyor.
- `boyut = orta` dalında `renk` ve `şekil` **eşit** kazanç verdi (0.971). Kod ilk sıradakini (`renk`) seçti. Gerçek kütüphanelerde de eşitlikler belli bir kurala göre (çoğu zaman sıra veya rastgelelik ile) bozulur. Bu yüzden `random_state` değişince ağaç değişebilir.
- Ağaç toplam iki soruyla 12 örneğin hepsini doğru sınıflandırıyor. Ama `orta + yeşil → elma` yaprağı **tek bir örneğe** dayanıyor. Bu, büyük ağaçlarda aşırı öğrenmenin nasıl başladığını gösterir.

## 6. Bilgi Kazancının Zaafı: Çok Değerli Özellikler

Veriye her satır için benzersiz bir `kimlik` sütunu (ör. barkod) ekleyelim. Bu sütun her satırı ayrı bir saf yaprağa böler, yani çocuk entropisi 0 olur ve kazanç **maksimum** çıkar. Oysa yeni bir meyve için bu sütun hiçbir şey söylemez.

C4.5 algoritması bunu **kazanç oranı** (*gain ratio*) ile düzeltir. Kazancı, bölmenin kendi entropisine (*SplitInfo*) böler:

$$
\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}, \qquad \text{SplitInfo}(S, A) = -\sum_v \frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}
$$

Önceki kodun devamı olarak:

```python

# Her satıra benzersiz bir "kimlik" sütunu ekleyelim (ör. ürün barkodu)
VERI_ID = [(f"id{i}",) + s for i, s in enumerate(VERI)]


def bolunme_bilgisi(satirlar, j):
    """SplitInfo = -Σ (|S_v|/|S|)·log2(|S_v|/|S|): bölmenin kendi entropisi."""
    return entropi([s[j] for s in satirlar])


for j, o in enumerate(["kimlik"] + OZELLIKLER):
    ig = bolme_kazanci(VERI_ID, j)
    si = bolunme_bilgisi(VERI_ID, j)
    print(f"{o:7s} kazanç = {ig:.3f}   SplitInfo = {si:.3f}   kazanç oranı = {ig / si:.3f}")
```

**Çıktı:**

```text
kimlik  kazanç = 2.522   SplitInfo = 3.585   kazanç oranı = 0.704
boyut   kazanç = 1.555   SplitInfo = 1.555   kazanç oranı = 1.000
renk    kazanç = 1.355   SplitInfo = 1.585   kazanç oranı = 0.855
şekil   kazanç = 0.645   SplitInfo = 0.811   kazanç oranı = 0.795
```

Saf bilgi kazancına bakılsaydı köke anlamsız `kimlik` sütunu gelirdi (2.522 > 1.555). Kazanç oranı ise `kimlik`'i 12 parçaya böldüğü için cezalandırır (SplitInfo = $\log_2 12 = 3.585$) ve `boyut`'u yeniden ilk sıraya koyar. CART (scikit-learn) ise her düğümde yalnızca **ikili** bölme yaptığı için bu sorundan daha az etkilenir. Yine de kimlik benzeri sütunlar modele hiç verilmemelidir.

## 7. Sayısal Özellikler ve Durma Kuralları

- **Sayısal özellikler:** Değerler sıralanır, ardışık iki değerin ortası aday eşik olur ("boyut ≤ 7.5 cm?"). Her eşik için kazanç hesaplanır ve en iyisi seçilir. Böylece sayısal bir özellik de ikili bir soruya dönüşür.
- **Durma (ön budama):** `max_depth`, `min_samples_leaf`, `min_samples_split`, `min_impurity_decrease`. Bunlar olmadan ağaç her yaprak saf olana kadar büyür ve eğitim verisini ezberler.
- **Sonradan budama:** Ağaç tam büyütülür, sonra doğrulama hatasını artırmayan dallar kesilir (scikit-learn'de `ccp_alpha`). Hangi değerin en iyi olduğunu seçmek için [çapraz doğrulama](2026-11-21-k-katli-capraz-dogrulama-ile-model-secimi.md) kullanılır.

scikit-learn ile aynı fikir:

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy",   # varsayılan: "gini"
                               max_depth=3, min_samples_leaf=5, random_state=0)
```

## 8. Sık Yapılan Hatalar

- ❌ **Çocuk safsızlıklarını ağırlıksız ortalamak.** 1 örneklik saf bir çocuk, 100 örneklik karışık bir çocukla aynı ağırlığı almamalı. Formüldeki $|S_v|/|S|$ çarpanı bu yüzden var.
- ❌ **Derinliği sınırlamadan ağaç eğitip eğitim doğruluğuna sevinmek.** Tam büyümüş ağacın eğitim doğruluğu neredeyse her zaman %100'dür. Bunun bir anlamı yoktur, başarıyı doğrulama verisiyle ölçün.
- ❌ **Kimlik, tarih-saat damgası veya hedefi sızdıran sütunları modele vermek.** Ağaç bunları büyük bir istekle köke yerleştirir (bkz. [Veri Sızıntısı notu](2026-11-18-veri-sizintisi-data-leakage.md)).
- ❌ **Ağaçlar için gereksiz ölçekleme yapmak.** Eşik tabanlı bölmeler ölçekten etkilenmez. Standartlaştırma ağaca zarar vermez ama fayda da sağlamaz (K-Means, gradyan inişi veya KNN'den farklı olarak).
- ❌ **`feature_importances_` değerlerini kesin gerçek sanmak.** Safsızlık tabanlı önem, çok değerli/sürekli özellikleri kayırır. Daha güvenilir bir ölçü için *permutation importance* kullanın.
- ❌ **Tek bir ağaca fazla güvenmek.** Veride küçük bir değişiklik bile kökteki soruyu ve dolayısıyla bütün ağacı değiştirebilir (yüksek varyans). Rastgele Orman (*Random Forest*) ve gradyan artırma (*gradient boosting*) bu kararsızlığı çok sayıda ağacı birleştirerek azaltır.

## 9. Mini Alıştırma

**Soru:** Bir düğümde 10 örnek var: 6 "evet", 4 "hayır". $A$ özelliği bu düğümü ikiye bölüyor: Sol çocuk 4 "evet" ve 0 "hayır", sağ çocuk 2 "evet" ve 4 "hayır" içeriyor.

1. Ebeveyn düğümün entropisini ve Gini safsızlığını hesaplayın.
2. $A$'ya göre bölmenin bilgi kazancı ve Gini azalması nedir?
3. Başka bir $B$ özelliği düğümü 5 "evet" + 0 "hayır" ve 1 "evet" + 4 "hayır" diye bölüyor. $A$ mı, $B$ mi seçilir?

<details>
<summary>Çözüm</summary>

1. $H = -(0.6\log_2 0.6 + 0.4\log_2 0.4) \approx 0.971$ bit. $G = 1 - 0.6^2 - 0.4^2 = 0.48$.
2. Sol çocuk saf: $H = 0$, $G = 0$. Sağ çocuk (2/6, 4/6): $H \approx 0.918$, $G = 1 - \tfrac{1}{9} - \tfrac{4}{9} \approx 0.444$.
   - $\text{IG} = 0.971 - \tfrac{6}{10}(0.918) \approx 0.971 - 0.551 = 0.420$ bit.
   - Gini azalması $= 0.48 - \tfrac{6}{10}(0.444) \approx 0.48 - 0.267 = 0.213$.
3. $B$ için sağ çocuk (1/5, 4/5): $H \approx 0.722$, $G = 1 - 0.04 - 0.64 = 0.32$.
   - $\text{IG}_B = 0.971 - \tfrac{5}{10}(0.722) \approx 0.610$ bit.
   - Gini azalması $= 0.48 - 0.5(0.32) = 0.32$.

   İki ölçüye göre de **$B$** daha iyi bölmedir ve seçilir.
</details>

---
*Özet:* Karar ağacı her düğümde, safsızlığı (entropi veya Gini) en çok azaltan soruyu açgözlü biçimde seçer. Bilgi kazancı, ebeveyn safsızlığından çocukların **ağırlıklı** safsızlığını çıkarır. Çok değerli özellikler bu ölçüyü kandırabilir: kazanç oranı kullanın veya böyle sütunları hiç vermeyin. Derinliği ve yaprak boyutunu sınırlamadan kurulan ağaç veriyi ezberler.
