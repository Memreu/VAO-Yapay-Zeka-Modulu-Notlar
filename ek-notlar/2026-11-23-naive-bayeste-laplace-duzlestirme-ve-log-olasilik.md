# Naive Bayes'te Laplace Düzleştirmesi ve Log-Olasılık Hilesi: Sıfırlar ve Sıfıra Yaklaşan Çarpımlar

> **Bağlantılı olduğu dersler:** Ders 8 – Naive Bayes (Bayes teoremi, *Bag of Words*, Laplace düzeltmesi) · Ders 4 – Sınıflandırma
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Hatırlatma: Naive Bayes Ne Hesaplar?

Ders 8'de bir metnin sınıfını Bayes teoremiyle bulduk. Paydadaki $P(\text{metin})$ bütün sınıflar için aynı olduğundan atılır. "Naive" (saf) bağımsızlık varsayımıyla metnin olabilirliği kelimelerin olabilirliklerinin çarpımına dönüşür:

$$
\hat{c} = \arg\max_c \; P(c) \prod_{i=1}^{n} P(w_i \mid c)
$$

Kâğıt üzerinde basit görünen bu formül, koda döküldüğünde iki pratik soruna yol açar:

1. **Sıfır olasılık sorunu:** Eğitimde bir sınıfta hiç geçmemiş tek bir kelime, $P(w \mid c) = 0$ demektir. Çarpımın tamamı sıfır olur ve diğer bütün kanıtlar yok sayılır.
2. **Alt taşma (*underflow*) sorunu:** Her $P(w \mid c)$ 1'den çok küçüktür (tipik olarak $10^{-2}$ ile $10^{-5}$ arası). Yüzlerce kelimelik bir e-postada çarpım, bilgisayarın gösterebileceği en küçük sayının ($\approx 10^{-308}$) altına iner ve `0.0` olur.

Birinci sorunu **Laplace düzleştirmesi**, ikincisini **logaritma** çözer. Bu not ikisini de sayılarla gösteriyor.

## 2. Laplace (Toplamsal) Düzleştirme

Çok terimli (*multinomial*) Naive Bayes'te kelime olasılığı sayımlardan tahmin edilir:

$$
P(w \mid c) = \frac{\text{sayı}(w, c) + \alpha}{\sum_{w'} \text{sayı}(w', c) + \alpha \cdot |V|}
$$

- $\text{sayı}(w, c)$: $w$ kelimesinin $c$ sınıfındaki belgelerde toplam kaç kez geçtiği
- $|V|$: sözlük boyutu (eğitimde görülen farklı kelime sayısı)
- $\alpha$: düzleştirme katsayısı. $\alpha = 1$ ise **Laplace**, $0 < \alpha < 1$ ise **Lidstone** düzleştirmesi denir.

**Sezgi:** Her kelimeyi her sınıfta "zaten $\alpha$ kez görmüşüz" gibi davranırız. Paydaya $\alpha |V|$ eklememizin nedeni, olasılıkların toplamının yine 1 olmasıdır. Böylece hiçbir kelimenin olasılığı tam sıfır olmaz. Az görülen kelimeler biraz yukarı, çok görülenler biraz aşağı çekilir.

## 3. Log-Olasılık Hilesi

Logaritma monoton artan bir fonksiyondur. Bu nedenle $\arg\max$'ı değiştirmez ve çarpımı toplama çevirir:

$$
\hat{c} = \arg\max_c \Big[ \log P(c) + \sum_{i=1}^{n} \log P(w_i \mid c) \Big]
$$

$10^{-500}$ gibi bir sayı `float` ile gösterilemez ama logaritması olan $-1151.3$ rahatça gösterilir.

Yalnızca sınıf tahmini gerekiyorsa en büyük skoru seçmek yeterlidir. Olasılık ($P(\text{spam} \mid \text{metin})$) gerekiyorsa skorları normalleştirmek gerekir. Bunu yaparken $e^{-1060}$'ı doğrudan hesaplamak yine sıfır verir. Çözüm **log-sum-exp hilesidir**: Önce en büyük skor $m$ çıkarılır, sonra üs alınır:

$$
P(c \mid \text{metin}) = \frac{e^{s_c - m}}{\sum_{k} e^{s_k - m}}, \qquad m = \max_k s_k
$$

Bu ifade cebirsel olarak orijinaliyle aynıdır, çünkü pay ve paydadaki $e^{-m}$ birbirini götürür. En büyük terim $e^0 = 1$ olur, bu yüzden payda hiçbir zaman sıfır çıkmaz. (Ders 3'teki Softmax'ın kararlı hesabında da aynı numara kullanılır.)

## 4. Çalışan Örnek: Sıfırdan Multinomial Naive Bayes

Dokuz e-postalık küçük bir spam veri seti üzerinde düzleştirmeli ve log tabanlı bir sınıflandırıcı yazalım. Ardından düzleştirme kapatıldığında ve uzun bir e-posta geldiğinde ne olduğuna bakalım.

```python
import math
from collections import Counter

egitim = [
    ("bedava hediye kazan hemen tikla", "spam"),
    ("hemen kazan bedava tatil", "spam"),
    ("indirim kodu bedava kargo tikla", "spam"),
    ("tebrikler odul kazandiniz hemen arayin", "spam"),
    ("yarin toplanti saat onda", "normal"),
    ("rapor ekte toplanti notlari", "normal"),
    ("aksam yemegi icin yarin musait misin", "normal"),
    ("proje raporu yarin teslim", "normal"),
    ("kargo yarin gelir mi", "normal"),
]


class CokTerimliNB:
    """Kelime sayımlarıyla çalışan, sıfırdan yazılmış Multinomial Naive Bayes."""

    def __init__(self, alfa=1.0):
        self.alfa = alfa  # Laplace / Lidstone düzleştirme katsayısı

    def fit(self, veri):
        self.siniflar = sorted({y for _, y in veri})
        self.sayim = {c: Counter() for c in self.siniflar}
        belge = Counter()
        for metin, y in veri:
            self.sayim[y].update(metin.split())
            belge[y] += 1
        self.sozluk = set().union(*self.sayim.values())
        self.V = len(self.sozluk)
        self.toplam = {c: sum(self.sayim[c].values()) for c in self.siniflar}
        self.log_prior = {c: math.log(belge[c] / len(veri)) for c in self.siniflar}
        return self

    def log_olabilirlik(self, kelime, c):
        pay = self.sayim[c][kelime] + self.alfa
        payda = self.toplam[c] + self.alfa * self.V
        return math.log(pay / payda) if pay > 0 else float("-inf")

    def skorlar(self, metin):
        # log P(c) + Σ log P(w|c)  — çarpım yerine toplam
        return {
            c: self.log_prior[c]
            + sum(self.log_olabilirlik(w, c) for w in metin.split() if w in self.sozluk)
            for c in self.siniflar
        }

    def predict_proba(self, metin):
        s = self.skorlar(metin)
        m = max(s.values())
        if m == float("-inf"):  # her sınıf imkânsız çıktıysa (alfa=0 durumu)
            return {c: 1 / len(s) for c in s}
        us = {c: math.exp(v - m) for c, v in s.items()}  # log-sum-exp hilesi
        Z = sum(us.values())
        return {c: v / Z for c, v in us.items()}


model = CokTerimliNB(alfa=1.0).fit(egitim)
print("Sözlük:", model.V, "| sınıftaki kelime sayısı:", model.toplam)
for w in ["bedava", "yarin", "kargo"]:
    print(f"P({w}|c) =", {c: round(math.exp(model.log_olabilirlik(w, c)), 4) for c in model.siniflar})

print("\n--- alfa=1 tahminleri ---")
testler = ["bedava kargo yarin", "toplanti raporu hemen tikla", "yarin bedava tatil kazan"]
for t in testler:
    p = model.predict_proba(t)
    print(f"{t:30} -> {max(p, key=p.get):6} P(spam)={p['spam']:.3f}")

print("\n--- alfa=0 (düzleştirme yok) ---")
m0 = CokTerimliNB(alfa=0.0).fit(egitim)
for t in testler:
    print(f"{t:30} -> skorlar={m0.skorlar(t)}")

print("\n--- Alt taşma (underflow) ---")
uzun = " ".join(["bedava kazan hemen tikla"] * 100)  # 400 kelimelik e-posta
duz = {}
for c in model.siniflar:
    p = math.exp(model.log_prior[c])
    for w in uzun.split():
        p *= math.exp(model.log_olabilirlik(w, c))
    duz[c] = p
print("Düz çarpım :", duz)
print("Log skor   :", {c: round(v, 1) for c, v in model.skorlar(uzun).items()})
print("P(spam)    :", model.predict_proba(uzun)["spam"])

print("\n--- alfa'nın etkisi: 'bedava tatil kazan' ---")
for a in [0.01, 0.1, 1, 5, 20, 100, 1000]:
    p = CokTerimliNB(a).fit(egitim).predict_proba("bedava tatil kazan")["spam"]
    print(f"alfa={a:<5} P(spam)={p:.3f}")
```

**Çıktı:**

```text
Sözlük: 30 | sınıftaki kelime sayısı: {'normal': 22, 'spam': 19}
P(bedava|c) = {'normal': 0.0192, 'spam': 0.0816}
P(yarin|c) = {'normal': 0.0962, 'spam': 0.0204}
P(kargo|c) = {'normal': 0.0385, 'spam': 0.0408}

--- alfa=1 tahminleri ---
bedava kargo yarin             -> normal P(spam)=0.433
toplanti raporu hemen tikla    -> spam   P(spam)=0.670
yarin bedava tatil kazan       -> spam   P(spam)=0.830

--- alfa=0 (düzleştirme yok) ---
bedava kargo yarin             -> skorlar={'normal': -inf, 'spam': -inf}
toplanti raporu hemen tikla    -> skorlar={'normal': -inf, 'spam': -inf}
yarin bedava tatil kazan       -> skorlar={'normal': -inf, 'spam': -inf}

--- Alt taşma (underflow) ---
Düz çarpım : {'normal': 0.0, 'spam': 0.0}
Log skor   : {'normal': -1581.1, 'spam': -1060.6}
P(spam)    : 1.0

--- alfa'nın etkisi: 'bedava tatil kazan' ---
alfa=0.01  P(spam)=1.000
alfa=0.1   P(spam)=1.000
alfa=1     P(spam)=0.958
alfa=5     P(spam)=0.694
alfa=20    P(spam)=0.519
alfa=100   P(spam)=0.460
alfa=1000  P(spam)=0.446
```

## 5. Çıktıyı Okumak

**Kelime olasılıkları.** `spam` sınıfında 19, `normal` sınıfında 22 kelime var. Sözlük boyutu 30 olduğundan paydalar sırasıyla $19 + 30 = 49$ ve $22 + 30 = 52$ olur.

- `bedava` spam'de 3 kez geçiyor: $P(\text{bedava} \mid \text{spam}) = \tfrac{3+1}{49} \approx 0.0816$.
- `bedava` normal e-postalarda hiç geçmiyor ama olasılığı sıfır değil: $\tfrac{0+1}{52} \approx 0.0192$. Düzleştirmenin işi budur.
- `kargo` her iki sınıfta da birer kez geçtiği için iki olasılık birbirine çok yakın ($0.0385$ ve $0.0408$). Bu kelime ayırt edici değildir.

**Düzleştirme kapatılınca ($\alpha = 0$).** Üç test cümlesinin üçünde de **iki sınıfın skoru birden** $-\infty$ çıkıyor. Örneğin `bedava kargo yarin` cümlesinde `bedava` normal e-postalarda, `yarin` ise spam'de hiç geçmemiş. Her sınıf için en az bir "imkânsız" kelime var. Model iki sınıfı birbirinden ayıramaz hâle gelir ve tahmin anlamsızlaşır (kodumuz bu durumda yazı tura atar). Gerçek veride sözlük binlerce kelimedir, bu yüzden bu durum istisna değil, kuraldır.

**Alt taşma.** 400 kelimelik e-postada düz çarpım iki sınıf için de tam `0.0` çıkıyor ve karar verilemiyor. Log skorlar ise $-1581.1$ ve $-1060.6$ gibi gayet makul sayılar. Aralarındaki fark yaklaşık 520 olduğundan model spam'den emin: $e^{-520} \approx 10^{-226}$.

**$\alpha$'nın etkisi.** `bedava tatil kazan` cümlesinde:

| $\alpha$ | P(spam) | Yorum |
|---|---|---|
| 0.01 – 0.1 | ≈ 1.000 | Sayımlara neredeyse körü körüne güvenir, aşırı emin |
| 1 | 0.958 | Klasik Laplace, makul |
| 5 – 20 | 0.69 – 0.52 | Kanıt giderek bastırılıyor |
| 100 – 1000 | 0.46 – 0.446 | Bütün kelimeler eşit olasılıklı hâle gelir, sonuç önsel olasılığa ($P(\text{spam}) = 4/9 \approx 0.444$) yakınsar |

Yani $\alpha$ bir **düzenlileştirme (regularization)** parametresidir. Küçük $\alpha$, varyansı yüksek ve aşırı emin bir model üretir (ezberleme). Büyük $\alpha$ ise yanlılığı yüksek bir model üretir: Her şey önsele çekilir. En iyi değer veriye bağlıdır ve [k-katlı çapraz doğrulama](2026-11-21-k-katli-capraz-dogrulama-ile-model-secimi.md) ile seçilmelidir.

## 6. scikit-learn Karşılığı

Ders 8'deki `MultinomialNB` bu iki hileyi zaten uygular:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(CountVectorizer(), MultinomialNB())
izgara = {"multinomialnb__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
arama = GridSearchCV(pipe, izgara, cv=5, scoring="f1_macro")
# arama.fit(metinler, etiketler); print(arama.best_params_)
```

- `alpha` parametresi bizim $\alpha$'mızdır (varsayılan `1.0`).
- `feature_log_prob_` ve `class_log_prior_` öznitelikleri olasılıkları **log** olarak saklar. `predict_log_proba` de normalleştirmeyi log-sum-exp ile yapar.
- `CountVectorizer` varsayılan olarak tek harfli kelimeleri atar ve küçük harfe çevirir. Bu yüzden sayılar elle yazdığımız sürümle birebir aynı çıkmayabilir.

## 7. Sık Yapılan Hatalar

- ❌ **`alpha=0` vermek veya düzleştirmeyi unutmak.** Tek bir görülmemiş kelime bütün kanıtı siler. scikit-learn sürümüne ve `force_alpha` ayarına göre ya uyarı verip `alpha`'yı çok küçük bir sayıya (`1e-10`) yükseltir ya da sıfırı olduğu gibi kullanıp $-\infty$ log olasılıklar üretir. Hiçbirine güvenmeyin, $\alpha > 0$ verin.
- ❌ **Olasılıkları çarpıp sonra logaritmasını almak.** `math.log(p1 * p2 * ...)` alt taşmayı engellemez, çünkü çarpım logaritma alınmadan önce zaten `0.0` olmuştur ve `log(0)` hata verir. Her terimin logaritmasını ayrı alıp toplayın.
- ❌ **Log skorları doğrudan `exp` ile olasılığa çevirmek.** $e^{-1060}$ sıfır çıkar ve $0/0$ hatası alınır. Önce en büyük skoru çıkarın (log-sum-exp).
- ❌ **Naive Bayes'in `predict_proba` çıktısını kalibre edilmiş olasılık sanmak.** Bağımsızlık varsayımı aynı kanıtı defalarca sayar ("bedava" 100 kez geçerse 100 ayrı kanıt gibi). Bu yüzden olasılıklar 0 veya 1'e aşırı yakın çıkar. Sıralama (hangisi daha spam) genelde iyidir ama mutlak değere güvenilecekse `CalibratedClassifierCV` kullanın.
- ❌ **Sözlüğü test verisini de katarak kurmak.** $|V|$ ve kelime sayımları yalnızca eğitim verisinden hesaplanmalıdır. Aksi hâlde [veri sızıntısı](2026-11-18-veri-sizintisi-data-leakage.md) oluşur. Testte karşılaşılan bilinmeyen kelimeler (yukarıdaki koddaki gibi) atlanır.
- ❌ **$\alpha$'yı test kümesine bakarak seçmek.** Diğer hiperparametreler gibi çapraz doğrulama ile seçilmelidir.

## 8. Mini Alıştırma

**Soru:** Yukarıdaki eğitim verisi ve $\alpha = 1$ ile `bedava yarin` e-postasını sınıflandırın.

1. $P(\text{bedava} \mid c)$ ve $P(\text{yarin} \mid c)$ olasılıklarını kesir olarak yazın. (`yarin` normal e-postalarda 4 kez geçer, spam'de hiç geçmez.)
2. Her sınıfın (normalleştirilmemiş) skorunu ve $P(\text{spam} \mid \text{bedava yarin})$ olasılığını hesaplayın.
3. Model hangi sınıfı seçer? `bedava` spam'e 4 kat daha fazla işaret ederken neden bu sonuç çıkıyor?

<details>
<summary>Çözüm</summary>

1. Spam paydası $19 + 30 = 49$, normal paydası $22 + 30 = 52$.
   - $P(\text{bedava} \mid \text{spam}) = \tfrac{3+1}{49} = \tfrac{4}{49}$, $\;P(\text{yarin} \mid \text{spam}) = \tfrac{0+1}{49} = \tfrac{1}{49}$
   - $P(\text{bedava} \mid \text{normal}) = \tfrac{0+1}{52} = \tfrac{1}{52}$, $\;P(\text{yarin} \mid \text{normal}) = \tfrac{4+1}{52} = \tfrac{5}{52}$
2. Skorlar:
   - spam: $\tfrac{4}{9} \cdot \tfrac{4}{49} \cdot \tfrac{1}{49} = \tfrac{16}{21609} \approx 7.40 \times 10^{-4}$ (log: $-7.208$)
   - normal: $\tfrac{5}{9} \cdot \tfrac{1}{52} \cdot \tfrac{5}{52} = \tfrac{25}{24336} \approx 1.03 \times 10^{-3}$ (log: $-6.881$)
   - $P(\text{spam} \mid \text{metin}) = \tfrac{7.40}{7.40 + 10.27} \approx 0.419$
3. Model **normal** sınıfını seçer. `bedava` spam lehine $\tfrac{4/49}{1/52} \approx 4.24$ katlık kanıt sunar. `yarin` ise normal lehine $\tfrac{5/52}{1/49} \approx 4.71$ katlık kanıt sunar ve önsel olasılık da normal sınıfı ($5/9$ ve $4/9$) destekler. İki kelimenin etkisi neredeyse birbirini götürür, terazinin dengesini önsel olasılık belirler. (Kodla kontrol: `model.predict_proba("bedava yarin")` → `{'normal': 0.581, 'spam': 0.419}`.)
</details>

---
*Özet:* Naive Bayes'in formülü çarpımdır, ama koddaki karşılığı **log olasılıkların toplamı** olmalıdır. Aksi hâlde uzun metinlerde sonuç `0.0`'a taşar. Görülmemiş kelimelerin bütün kanıtı silmesini **Laplace ($\alpha$) düzleştirmesi** önler. $\alpha$ bir düzenlileştirme parametresidir: Küçükken model aşırı emin olur, büyüdükçe tahmin önsel olasılığa yakınsar. Olasılık gerekiyorsa normalleştirmeyi log-sum-exp ile yapın.
