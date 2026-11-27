# Geri Yayılımı (Backpropagation) Sıfırdan Yazmak: XOR, Zincir Kuralı ve Gradyan Kontrolü

> **Bağlantılı olduğu dersler:** Ders 1 – Bölüm 3: Yapay Sinir Ağları, Geri Yayılım Algoritması (1986), İleri Besleme / Hata Hesaplama / Ağırlık Güncelleme · Ders 5 – Bölüm 1: Ayrılabilirlik Analizi
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Hatırlatma: Ağ Nasıl Öğreniyordu?

Ders 1'de bir ağın öğrenme döngüsünü üç adımda görmüştük:

1. **İleri besleme (forward pass):** Girdi katman katman ilerler, ağ bir $y$ tahmini üretir.
2. **Hata hesaplama:** Tahmin $y$ ile istenen değer $y^D$ karşılaştırılır.
3. **Geri yayılım ve güncelleme:** Hata çıktıdan girdiye doğru "geri" taşınır ve her ağırlık, hatayı azaltacak yönde biraz değiştirilir.

Üçüncü adım çoğu zaman bir kara kutu olarak kalır: "Hatayı geri taşımak" tam olarak ne demek? Bu notta küçük bir ağı kütüphanesiz yazıp bu soruyu cevaplayacağız. Cevap kısaca şu: geri yayılım, **zincir kuralının** akıllıca düzenlenmiş hâlidir. Her ağırlığın kayba etkisini, ($\partial L / \partial w$), tek bir geri geçişte hesaplar. Bulunan gradyanlarla yapılan güncelleme ise [gradyan inişi notunda](2026-11-20-gradyan-inisi-ve-ogrenme-orani.md) gördüğümüz $w \leftarrow w - \eta\,\partial L/\partial w$ adımının aynısıdır.

## 2. Problem: XOR ve Tek Nöronun Sınırı

| $x_1$ | $x_2$ | XOR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Tek bir nöron $\sigma(w_1x_1 + w_2x_2 + b)$ düzlemi **tek bir doğruyla** ikiye böler. XOR'un 1'leri köşegenin bir tarafında, 0'ları öbür köşegende durur. Bu yüzden onları ayıran tek bir doğru yoktur, yani problem doğrusal olarak ayrılamaz (Ders 5'teki ayrılabilirlik tartışması). 1969'da Minsky ve Papert'in dikkat çektiği bu sınırlama, ilk yapay zekâ kışının sebeplerinden biri sayılır. Çözüm araya bir **gizli katman** koymaktır. Gizli katmanı eğitmenin pratik yolu da 1986'da yaygınlaşan geri yayılımdır.

Kuracağımız ağ:

$$
\underbrace{x \in \mathbb{R}^2}_{\text{girdi}}
\;\xrightarrow{\;W_1,\,b_1\;}\;
\underbrace{h = \sigma(W_1x + b_1) \in \mathbb{R}^H}_{\text{gizli katman}}
\;\xrightarrow{\;W_2,\,b_2\;}\;
\underbrace{y = \sigma(W_2 \cdot h + b_2)}_{\text{çıktı, } P(\text{XOR}=1)}
$$

Kayıp olarak ikili çapraz entropi (log-loss) kullanıyoruz: $L = -\big[t\log y + (1-t)\log(1-y)\big]$, burada $t$ hedef etikettir.

## 3. Zincir Kuralı: Dört Satırlık Türetme

$z_2 = W_2\cdot h + b_2$ ve $z_{1,j} = W_{1,j}\cdot x + b_{1,j}$ ön-aktivasyonlar olsun. Kullanacağımız iki gerçek:

- Sigmoid'in türevi: $\sigma'(z) = \sigma(z)\,(1-\sigma(z))$
- Sigmoid ile log-loss birlikte kullanılınca çok güzel bir sadeleşme olur: $\dfrac{\partial L}{\partial z_2} = y - t$

Bunları zincirleyince:

$$
\begin{aligned}
\delta_2 &= \frac{\partial L}{\partial z_2} = y - t
&&\text{(çıktı hatası)}\\[4pt]
\frac{\partial L}{\partial W_{2,j}} &= \delta_2 \, h_j, \qquad \frac{\partial L}{\partial b_2} = \delta_2
&&\text{(çıktı katmanı ağırlıkları)}\\[4pt]
\delta_{1,j} &= \frac{\partial L}{\partial z_{1,j}} = \delta_2 \, W_{2,j}\, h_j(1-h_j)
&&\text{(hata, gizli katmana geri taşındı)}\\[4pt]
\frac{\partial L}{\partial W_{1,jk}} &= \delta_{1,j}\, x_k, \qquad \frac{\partial L}{\partial b_{1,j}} = \delta_{1,j}
&&\text{(gizli katman ağırlıkları)}
\end{aligned}
$$

**"Hatayı geri taşımak"** üçüncü satırdır. Gizli nöron $j$'nin hatası, çıktı hatasının o nörona bağlanan ağırlıkla ($W_{2,j}$) ve nöronun kendi eğimiyle ($h_j(1-h_j)$) çarpımıdır. Daha derin ağlarda aynı satır her katman için tekrarlanır. Genel kalıp hep aynıdır: **gradyan = (gelen δ) × (o ağırlığa giren değer)**.

## 4. Elle Bir Adım: Sayılarla Zincir Kuralı

Tek gizli nöronlu ($H=1$) minicik bir ağ alalım: $W_1 = [0.5,\,-0.5]$, $b_1 = 0$, $W_2 = 1$, $b_2 = 0$. Örnek $x = (1, 0)$, hedef $t = 1$.

| Adım | Hesap | Değer |
|---|---|---|
| $z_1$ | $0.5\cdot1 + (-0.5)\cdot 0 + 0$ | $0.5$ |
| $h$ | $\sigma(0.5)$ | $0.6225$ |
| $z_2$ | $1 \cdot 0.6225 + 0$ | $0.6225$ |
| $y$ | $\sigma(0.6225)$ | $0.6508$ |
| $L$ | $-\log(0.6508)$ | $0.4296$ |
| $\delta_2$ | $y - t = 0.6508 - 1$ | $-0.3492$ |
| $\partial L/\partial W_2$ | $\delta_2 \cdot h = -0.3492 \cdot 0.6225$ | $-0.2174$ |
| $\delta_1$ | $\delta_2 \cdot W_2 \cdot h(1-h) = -0.3492 \cdot 1 \cdot 0.2350$ | $-0.0821$ |
| $\partial L/\partial W_{1,1}$ | $\delta_1 \cdot x_1$ | $-0.0821$ |
| $\partial L/\partial W_{1,2}$ | $\delta_1 \cdot x_2 = \delta_1 \cdot 0$ | $0$ |

Gözlemler:

- Gradyanların hepsi negatif. Yani ağırlıkları **artırmak** kaybı azaltır. Bu mantıklı, çünkü $y$'nin 1'e yaklaşması gerekiyor.
- $x_2 = 0$ olduğu için $W_{1,2}$ bu örnekten **hiçbir şey öğrenmez**. Sıfır olan bir girdinin ağırlığı o örnekte güncellenmez.
- Gizli katmanın gradyanı ($-0.082$), çıktı katmanınınkinden ($-0.217$) küçüktür. Sigmoid'in eğimi en fazla $0.25$ olduğundan, hata her katmandan geçerken küçülür. Derin ağlarda bu etki **kaybolan gradyan** (*vanishing gradient*) sorununa yol açar.

$\eta = 1$ ile bir adım güncellersek ($W_2 = 1.2174$, $W_{1,1} = 0.5821$, $b_1 = 0.0821$, $b_2 = 0.3492$), aynı örnek için tahmin $0.6508 \to 0.7600$ olur, kayıp da $0.4296 \to 0.2744$'e düşer.

## 5. Çalışan Örnek: XOR'u Öğrenen Ağ

```python
import math, random

# --- 1) Veri: XOR. Tek bir nöron (doğrusal sınır) bunu çözemez ---
X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

# --- 2) Ağ: 2 girdi -> H gizli nöron (sigmoid) -> 1 çıktı (sigmoid) ---
def agi_kur(H, tohum):
    rnd = random.Random(tohum)
    return {
        "W1": [[rnd.uniform(-1, 1) for _ in range(2)] for _ in range(H)],
        "b1": [0.0] * H,
        "W2": [rnd.uniform(-1, 1) for _ in range(H)],
        "b2": 0.0,
    }

def ileri(p, x):
    """İleri besleme: ara değerleri de döndürür (geri yayılımda lazım)."""
    h = [sigmoid(sum(w * xi for w, xi in zip(p["W1"][j], x)) + p["b1"][j])
         for j in range(len(p["W2"]))]
    y = sigmoid(sum(w * hj for w, hj in zip(p["W2"], h)) + p["b2"])
    return h, y

def kayip(p):
    """Ortalama ikili çapraz entropi (log-loss)."""
    eps = 1e-12
    top = 0.0
    for x, t in zip(X, Y):
        _, y = ileri(p, x)
        top += -(t * math.log(y + eps) + (1 - t) * math.log(1 - y + eps))
    return top / len(X)

def geri(p):
    """Geri yayılım: zincir kuralıyla tüm gradyanları hesaplar."""
    H = len(p["W2"])
    g = {"W1": [[0.0, 0.0] for _ in range(H)], "b1": [0.0] * H,
         "W2": [0.0] * H, "b2": 0.0}
    m = len(X)
    for x, t in zip(X, Y):
        h, y = ileri(p, x)
        d2 = (y - t) / m                      # dL/dz2  (sigmoid + log-loss sadeleşmesi)
        for j in range(H):
            g["W2"][j] += d2 * h[j]           # dL/dW2_j = d2 * h_j
        g["b2"] += d2
        for j in range(H):
            d1 = d2 * p["W2"][j] * h[j] * (1 - h[j])   # hatayı geriye taşı
            g["W1"][j][0] += d1 * x[0]
            g["W1"][j][1] += d1 * x[1]
            g["b1"][j] += d1
    return g

# --- 3) Gradyan kontrolü: analitik türev = sayısal türev mi? ---
def gradyan_kontrolu(p, eps=1e-5):
    g = geri(p)
    en_kotu = 0.0
    def dene(oku, yaz, analitik):
        nonlocal en_kotu
        eski = oku()
        yaz(eski + eps); Lp = kayip(p)
        yaz(eski - eps); Lm = kayip(p)
        yaz(eski)
        sayisal = (Lp - Lm) / (2 * eps)
        fark = abs(sayisal - analitik) / max(1e-12, abs(sayisal) + abs(analitik))
        en_kotu = max(en_kotu, fark)
    H = len(p["W2"])
    for j in range(H):
        for k in range(2):
            dene(lambda: p["W1"][j][k], lambda v: p["W1"][j].__setitem__(k, v), g["W1"][j][k])
        dene(lambda: p["b1"][j], lambda v: p["b1"].__setitem__(j, v), g["b1"][j])
        dene(lambda: p["W2"][j], lambda v: p["W2"].__setitem__(j, v), g["W2"][j])
    dene(lambda: p["b2"], lambda v: p.__setitem__("b2", v), g["b2"])
    return en_kotu

# --- 4) Eğitim: gradyan inişi ---
def egit(H, tohum, lr=2.0, tur=5000, yazdir=False):
    p = agi_kur(H, tohum)
    for t in range(tur + 1):
        if yazdir and t in (0, 100, 500, 1000, 2000, 5000):
            print(f"  tur {t:5d}  kayıp = {kayip(p):.4f}")
        g = geri(p)
        for j in range(H):
            for k in range(2):
                p["W1"][j][k] -= lr * g["W1"][j][k]
            p["b1"][j] -= lr * g["b1"][j]
            p["W2"][j] -= lr * g["W2"][j]
        p["b2"] -= lr * g["b2"]
    return p

p0 = agi_kur(3, tohum=1)
print(f"Gradyan kontrolü (en büyük göreli fark): {gradyan_kontrolu(p0):.2e}")

print("\nEğitim (H=3 gizli nöron):")
p = egit(3, tohum=1, yazdir=True)
for x, t in zip(X, Y):
    print(f"  {x} -> {ileri(p, x)[1]:.3f}  (hedef {t})")

print("\nGizli nöron sayısı ve başlangıç tohumunun etkisi (10 tohum):")
for H in (1, 2, 3, 4):
    basari = sum(
        all((ileri(pp, x)[1] > 0.5) == (t == 1) for x, t in zip(X, Y))
        for pp in [egit(H, s) for s in range(10)])
    print(f"  H={H}: {basari}/10 denemede XOR çözüldü")
```

**Çıktı:**

```
Gradyan kontrolü (en büyük göreli fark): 2.98e-09

Eğitim (H=3 gizli nöron):
  tur     0  kayıp = 0.6934
  tur   100  kayıp = 0.6732
  tur   500  kayıp = 0.0143
  tur  1000  kayıp = 0.0054
  tur  2000  kayıp = 0.0024
  tur  5000  kayıp = 0.0008
  (0, 0) -> 0.001  (hedef 0)
  (0, 1) -> 0.999  (hedef 1)
  (1, 0) -> 0.999  (hedef 1)
  (1, 1) -> 0.001  (hedef 0)

Gizli nöron sayısı ve başlangıç tohumunun etkisi (10 tohum):
  H=1: 0/10 denemede XOR çözüldü
  H=2: 9/10 denemede XOR çözüldü
  H=3: 10/10 denemede XOR çözüldü
  H=4: 10/10 denemede XOR çözüldü
```

### Çıktıyı okumak

- **Gradyan kontrolü $3\cdot10^{-9}$:** Elle türettiğimiz formüller, kaybı her parametrede $\pm\varepsilon$ oynatarak bulunan sayısal türevle ($\frac{L(w+\varepsilon) - L(w-\varepsilon)}{2\varepsilon}$) dokuz basamak uyuşuyor. Kabaca $10^{-7}$'nin altı "doğru", $10^{-3}$'ün üstü ise "kesin hata var" demektir.
- **Başlangıçtaki $0.6934 \approx \ln 2$:** Ağ başta her şeye ~0.5 diyor, yani yazı tura atıyor.
- **İlk 100 turdaki plato:** Kayıp ilk 100 turda neredeyse hiç düşmüyor ($0.693 \to 0.673$), sonra 500. tura kadar hızla çöküyor. Gizli nöronların "kim hangi bölgeye bakacak" diye ayrışması zaman alır. Kayıp eğrisinde başta görülen düzlük, ağın öğrenmediği anlamına gelmez.
- **$H=1$ hiçbir zaman çözemiyor:** Tek gizli nöronlu ağ yine tek bir doğrusal sınıra bağlıdır. Yani gizli katmanı eklemek tek başına yetmez, katmanda **yeterli genişlik** de olmalıdır.
- **$H=2$ teoride yeter ama 10 denemeden biri başarısız:** O denemede ağ, yerel bir minimuma ya da eyer noktasına takılır. Gizli nöron sayısını biraz artırmak, kayıp yüzeyini "kolay" hâle getirir. Pratikte ağların gerekenden geniş kurulmasının bir sebebi de budur.

## 6. Sık Yapılan Hatalar

**1. Ağırlıkları aynı değerle (ör. hepsini sıfır) başlatmak.** Bütün gizli nöronlar aynı başlarsa aynı gradyanı alırlar ve sonsuza kadar birbirinin kopyası olarak kalırlar. Bu durumda 4 nöron, fiilen 1 nörona iner. Ana kodun sonuna eklenerek denenebilir:

```python
p = agi_kur(4, 0)
for j in range(4):
    p["W1"][j] = [0.5, 0.5]; p["W2"][j] = 0.5      # simetrik başlatma
for _ in range(5000):
    g = geri(p)
    for j in range(4):
        for k in range(2): p["W1"][j][k] -= 2.0 * g["W1"][j][k]
        p["b1"][j] -= 2.0 * g["b1"][j]; p["W2"][j] -= 2.0 * g["W2"][j]
    p["b2"] -= 2.0 * g["b2"]
print([[round(w, 3) for w in r] for r in p["W1"]])
print(round(kayip(p), 4), [round(ileri(p, x)[1], 3) for x in X])
```

```
[[8.014, 8.014], [8.014, 8.014], [8.014, 8.014], [8.014, 8.014]]
0.4789 [0.002, 0.666, 0.666, 0.667]
```

5000 turdan sonra dört satır hâlâ birebir aynı ve ağ (1,1) için yanlış cevap veriyor ($0.667 > 0.5$). Rastgele başlatmanın amacı bu **simetriyi kırmaktır**.

**2. Aktivasyonun türevini unutmak.** Geri taşıma satırında `h[j] * (1 - h[j])` çarpanı yazılmazsa kod hata vermeden çalışır, sadece yanlış yöne gider. Gradyan kontrolü bunu hemen yakalar:

```python
def geri_hatali(p):
    H = len(p["W2"])
    g = {"W1": [[0.0, 0.0] for _ in range(H)], "b1": [0.0] * H, "W2": [0.0] * H, "b2": 0.0}
    for x, t in zip(X, Y):
        h, y = ileri(p, x); d2 = (y - t) / len(X)
        for j in range(H): g["W2"][j] += d2 * h[j]
        g["b2"] += d2
        for j in range(H):
            d1 = d2 * p["W2"][j]                    # HATA: h(1-h) unutuldu
            g["W1"][j][0] += d1 * x[0]; g["W1"][j][1] += d1 * x[1]; g["b1"][j] += d1
    return g

geri = geri_hatali                                  # kontrol fonksiyonu artık bunu kullanır
print(f"{gradyan_kontrolu(agi_kur(3, 1)):.2e}")     # -> 6.17e-01
```

$0.617$, beklenen $10^{-9}$ civarından sekiz mertebe büyük. Kendi yazdığınız her geri yayılım kodunu eğitimden önce bu kontrolden geçirmek iyi bir alışkanlıktır.

**3. Ön-aktivasyonu ($z$) ve aktivasyonu ($h$) karıştırmak.** Sigmoid'in türevi $h(1-h)$ biçiminde yazılır. Burada $h = \sigma(z)$'dir, $z$ değil. `z * (1 - z)` yazmak yaygın ve sessiz bir hatadır. `ileri()` fonksiyonunun ara değerleri döndürmesinin sebebi de budur.

**4. Kaybı ve $\delta_2$'yi uyumsuz seçmek.** $\delta_2 = y - t$ sadeleşmesi yalnızca **sigmoid + log-loss** (ya da softmax + çapraz entropi) ikilisi için geçerlidir. Kayıp MSE olursa $\delta_2 = (y - t)\,y(1-y)$ olur. Bu durumda tahmin yanlış ama "eminken" ($y \approx 0$ veya $1$) gradyan neredeyse sıfıra iner ve öğrenme durur. Sınıflandırmada log-loss'un tercih edilmesinin bir sebebi budur.

**5. Öğrenme oranını her ağda aynı sanmak.** Buradaki $\eta = 2$, 4 örneklik ve kayıp ortalaması alınmış küçük bir problem için iyi çalışıyor. Başka bir veri ve mimaride aynı değer patlamaya ya da sürünmeye yol açabilir (bkz. [gradyan inişi notu](2026-11-20-gradyan-inisi-ve-ogrenme-orani.md)).

## 7. Kütüphane Karşılığı

Pratikte gradyanları elle yazmayız. PyTorch ve TensorFlow gibi kütüphaneler bu notta yaptığımız işi (ileri geçişte ara değerleri saklayıp geri geçişte zincir kuralını uygulamak) **otomatik türev** (*autograd*) ile her mimari için kendiliğinden yapar. scikit-learn'de aynı ağın karşılığı kabaca şudur:

```python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(3,), activation="logistic",
                      solver="sgd", learning_rate_init=0.5, max_iter=5000, random_state=1)
model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
```

Kütüphanenin içinde dönen şey yine bu nottaki `ileri()` → `geri()` → güncelleme döngüsüdür. Ancak ilk değer atama, mini-batch ve durma kuralları farklı olduğu için sonuçlar birebir aynı çıkmaz.

## 8. Alıştırma

Gizli katmandaki sigmoid'i $\tanh$ ile değiştirin (çıktı katmanı sigmoid kalsın).

**a)** `ileri()` ve `geri()` fonksiyonlarında hangi satırlar değişmeli? ($\tanh'(z) = 1 - \tanh^2(z)$)
**b)** Değişiklikten sonra gradyan kontrolü hâlâ küçük bir değer veriyor mu?
**c)** Eğitim eğrisini sigmoid'li hâliyle karşılaştırın. İlk 100 turdaki plato ne oldu, neden?

<details>
<summary><b>Çözüm</b></summary>

**a)** İki satır değişir:

```python
# ileri():
h = [math.tanh(sum(w * xi for w, xi in zip(p["W1"][j], x)) + p["b1"][j])
     for j in range(len(p["W2"]))]
# geri():
d1 = d2 * p["W2"][j] * (1 - h[j] ** 2)
```

**b)** Evet. Göreli fark yaklaşık $4.8\cdot10^{-9}$ çıkar, yani türetme doğru.

**c)** Çıktı:

```
  tur     0  kayıp = 0.6938
  tur   100  kayıp = 0.0156
  tur   500  kayıp = 0.0026
  ...
  H=1: 0/10, H=2: 9/10, H=3: 10/10, H=4: 10/10
```

Plato kaybolur: sigmoid'li ağın ancak 500. turda indiği kayıp düzeyine ($0.014$) tanh'lı ağ 100. turda neredeyse ulaşır ($0.016$). Bunun iki sebebi var:
1. $\tanh$'ın sıfırdaki eğimi $1$'dir, sigmoid'inki ise $0.25$. Böylece gizli katmana geri taşınan hata 4 kat daha büyük olur.
2. $\tanh$'ın çıktısı sıfır merkezlidir ($-1$ ile $1$ arası). Sigmoid'in çıktısı ise hep pozitiftir ($0$ ile $1$ arası). Bu yüzden sigmoid'de bir sonraki katmanın ağırlıklarına giden gradyanların hepsi aynı işareti taşır ve iniş zikzak çizer.

Kapasite değişmez: $H=1$ yine çözemez, çünkü sorun aktivasyonda değil, tek nöronun tek sınır çizebilmesindedir. Modern derin ağlarda gizli katmanlarda çoğunlukla ReLU ($\max(0, z)$, türevi $0$ ya da $1$) tercih edilir. Bunun sebebi de aynı: türev ne kadar az küçülürse hata o kadar derine taşınabilir.

</details>
