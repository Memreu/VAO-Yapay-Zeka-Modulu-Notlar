# Gradyan İnişi ve Öğrenme Oranı: Neden Bazen Yakınsar, Bazen Patlar?

> **Bağlantılı olduğu dersler:** Ders 2 – Denetimli Öğrenme (Curve Fitting) · Ders 4–5 – Düzenlileştirme ve Gradyan İnişi ile optimizasyon · Ders 3 – Sinir ağlarında gradyan sorunları
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Fikir: Yokuş Aşağı Küçük Adımlar

Bir modeli eğitmek, bir **kayıp fonksiyonunu** $L(\theta)$ en küçük yapan parametreleri $\theta$ bulmaktır. Gradyan $\nabla L(\theta)$, kaybın **en hızlı arttığı** yönü gösterir; biz tam ters yöne küçük bir adım atarız:

$$\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta_t)$$

| Sembol | Anlamı |
|---|---|
| $\theta$ | Parametreler (ör. doğrusal regresyonda $w$ ve $b$) |
| $\eta$ (öğrenme oranı, *learning rate*) | Adım büyüklüğü — bu notun asıl konusu |
| $\nabla L$ | Kaybın parametrelere göre kısmi türevleri |

Doğrusal regresyonda kapalı form çözüm (en küçük kareler) zaten vardır; ama sinir ağları, lojistik regresyon veya Lasso gibi modellerde böyle bir formül yoktur. Bu yüzden gradyan inişi yapay zekânın "iş atı" optimizasyon yöntemidir.

## 2. Doğrusal Regresyon İçin Gradyan

Model $\hat{y} = w x + b$, kayıp ortalama kare hata (MSE):

$$L(w,b) = \frac{1}{N}\sum_{i=1}^{N} (w x_i + b - y_i)^2$$

Hata $e_i = w x_i + b - y_i$ olmak üzere:

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum e_i\, x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{N}\sum e_i$$

Sezgi: Tahminler çoğunlukla büyükse ($e_i > 0$) türev pozitiftir ve güncelleme $w$ ile $b$'yi küçültür.

## 3. Çalışan Örnek: Dört Farklı Öğrenme Oranı

Veri $y = 4 + 3x + \text{gürültü}$ kuralıyla üretiliyor. Aynı başlangıçtan ($w = b = 0$) dört farklı $\eta$ ile 200 adım atıp sonucu kapalı form çözümle karşılaştırıyoruz. Sonda, hangi $\eta$ değerinin üstünde algoritmanın **patlayacağını** önceden hesaplıyoruz.

```python
import random

random.seed(42)

# Sentetik veri: y = 4 + 3x + gürültü,  x ∈ [0, 2]
N = 100
X = [random.uniform(0, 2) for _ in range(N)]
Y = [4 + 3 * x + random.gauss(0, 0.5) for x in X]


def mse(w, b):
    return sum((w * x + b - y) ** 2 for x, y in zip(X, Y)) / N


def gradyan(w, b):
    # L = (1/N) Σ (w·x + b − y)²  →  ∂L/∂w = (2/N) Σ e·x ,  ∂L/∂b = (2/N) Σ e
    dw = db = 0.0
    for x, y in zip(X, Y):
        e = w * x + b - y
        dw += 2 * e * x / N
        db += 2 * e / N
    return dw, db


def gradyan_inisi(lr, adim=200):
    w, b = 0.0, 0.0
    gecmis = []
    for t in range(adim + 1):
        if t in (0, 10, 50, 200):
            gecmis.append((t, mse(w, b)))
        dw, db = gradyan(w, b)
        w -= lr * dw
        b -= lr * db
    return w, b, gecmis


# Kapalı form (en küçük kareler) çözümü — referans
mx, my = sum(X) / N, sum(Y) / N
w_ols = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
b_ols = my - w_ols * mx
print(f"Kapalı form (OLS): w = {w_ols:.4f}, b = {b_ols:.4f}, MSE = {mse(w_ols, b_ols):.4f}\n")

print(f"{'lr':>6} | {'MSE t=0':>9} {'t=10':>9} {'t=50':>9} {'t=200':>11} | {'w':>8} {'b':>8}")
for lr in (0.001, 0.05, 0.3, 0.6):
    w, b, g = gradyan_inisi(lr)
    kayip = " ".join(f"{m:9.4f}" if m < 1e6 else f"{m:9.2e}" for _, m in g[:3])
    son = g[3][1]
    son_s = f"{son:11.4f}" if son < 1e6 else f"{son:11.2e}"
    wb = f"{w:8.3f} {b:8.3f}" if abs(w) < 1e6 else f"{'ıraksadı':>17}"
    print(f"{lr:>6} | {kayip} {son_s} | {wb}")

# Kararlılık sınırı: MSE'nin Hessian'ı H = (2/N)·[[Σx², Σx], [Σx, N]] sabittir.
# Gradyan inişi ancak lr < 2 / λ_max(H) ise yakınsar.
a = 2 * sum(x * x for x in X) / N
c = 2 * sum(X) / N
d = 2.0
iz, det = a + d, a * d - c * c
l_max = iz / 2 + ((iz / 2) ** 2 - det) ** 0.5
l_min = iz / 2 - ((iz / 2) ** 2 - det) ** 0.5
print(f"\nλ_max = {l_max:.3f}, λ_min = {l_min:.3f}, koşul sayısı = {l_max / l_min:.1f}")
print(f"Kararlılık sınırı: lr < 2/λ_max = {2 / l_max:.3f}")
```

**Çıktı:**

```
Kapalı form (OLS): w = 3.0945, b = 3.9963, MSE = 0.2278

    lr |   MSE t=0      t=10      t=50       t=200 |        w        b
 0.001 |   52.0313   47.8598   34.2806      9.9534 |    2.093    1.907
  0.05 |   52.0313    0.8032    0.2579      0.2280 |    3.118    3.970
   0.3 |   52.0313    0.2476    0.2278      0.2278 |    3.095    3.996
   0.6 |   52.0313 225619.4385  8.22e+19    3.31e+74 |          ıraksadı

λ_max = 4.201, λ_min = 0.328, koşul sayısı = 12.8
Kararlılık sınırı: lr < 2/λ_max = 0.476
```

### Çıktıyı Okumak

- **$\eta = 0.001$ (çok küçük):** Kayıp düzenli azalıyor ama 200 adımda hâlâ 9.95; $w = 2.09$, $b = 1.91$ gerçek değerlerden çok uzak. Algoritma doğru yönde ama **çok yavaş**.
- **$\eta = 0.05$ (makul):** 200 adımda MSE 0.2280'e iniyor; çözüm kapalı forma oldukça yakın.
- **$\eta = 0.3$ (iyi):** 50 adımda kapalı form çözüme (MSE 0.2278, $w = 3.095$, $b = 3.996$) ulaşıyor.
- **$\eta = 0.6$ (çok büyük):** Her adım vadinin karşı yamacına, öncekinden daha yükseğe sıçrıyor; 10 adımda kayıp 52'den 225 bine, 200 adımda $10^{74}$'e çıkıyor → **ıraksama**.

Sınır şans eseri değil, hesaplanabilir: MSE'nin ikinci türev matrisi (Hessian) $H$ sabittir ve gradyan inişi ancak

$$\eta < \frac{2}{\lambda_{\max}(H)}$$

ise yakınsar. Burada $\lambda_{\max} = 4.201$ → sınır $\eta < 0.476$. 0.3 sınırın altında kaldığı için çalıştı, 0.6 üstünde kaldığı için patladı.

## 4. Koşul Sayısı ve Özellik Ölçekleme

Yakınsama hızını en "düz" yön, yani $\lambda_{\min}$ belirler; ama $\eta$'yı en "dik" yön ($\lambda_{\max}$) sınırlar. Oran

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \quad \text{(koşul sayısı)}$$

büyüdükçe kayıp yüzeyi uzun, dar bir vadiye dönüşür: Dik yönde zikzak çizerken düz yönde neredeyse hiç ilerlenemez. Örneğimizde $\kappa = 12.8$.

Özellikler farklı ölçeklerdeyse (ör. yaş: 20–70, gelir: 10 000–200 000) $\kappa$ binlerle ifade edilir. **Standartlaştırma** ($x' = (x - \mu)/\sigma$) Hessian'ı köşegene yaklaştırır, $\kappa$'yı düşürür ve hem daha büyük $\eta$ kullanmaya hem de çok daha hızlı yakınsamaya izin verir. Ders 3'teki EDA ve ön işleme adımlarında ölçeklemenin önemi bu yüzden vurgulanır. (Ölçekleyiciyi **yalnızca eğitim verisine** fit etmeyi unutma — bkz. [Veri Sızıntısı notu](2026-11-18-veri-sizintisi-data-leakage.md).)

## 5. Batch, Stokastik ve Mini-Batch

| Yöntem | Bir adımda kullanılan veri | Artı | Eksi |
|---|---|---|---|
| **Batch GD** (yukarıdaki kod) | Tüm $N$ örnek | Pürüzsüz, kararlı iniş | Büyük veride her adım pahalı |
| **Stokastik GD (SGD)** | 1 örnek | Çok ucuz adımlar, yerel minimumlardan kaçabilir | Gürültülü; sonda $\eta$ azaltılmalı |
| **Mini-batch** | 32–512 örnek | GPU ile verimli, dengeli gürültü | Batch boyutu da bir hiperparametre |

Derin öğrenmede pratikte mini-batch kullanılır; Adam, RMSProp gibi optimizasyon algoritmaları her parametre için adım büyüklüğünü uyarlayarak koşul sayısı sorununu da kısmen hafifletir.

## 6. Sık Yapılan Hatalar

- ❌ **Kayıp eğrisini çizmemek.** Kaybın adımlara göre grafiği en ucuz teşhis aracıdır: Yavaş ve düz iniş → $\eta$ küçük; zikzak / artan kayıp / `nan` → $\eta$ büyük.
- ❌ **Kayıp `nan` olunca veriden şüphelenmek.** Çoğu zaman sebep ıraksamadır; önce $\eta$'yı 3–10 kat düşür.
- ❌ **Özellikleri ölçeklememek.** Ölçek farkı koşul sayısını büyütür; "hiçbir $\eta$ iyi çalışmıyor" hissinin en sık sebebidir.
- ❌ **$\eta$'yı doğrusal aralıkta aramak.** 0.1, 0.2, 0.3 … yerine **logaritmik** ızgara dene: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3.
- ❌ **Gradyanı yanlış türetmek.** Elle yazılan türevleri küçük bir $h$ ile sayısal türevle karşılaştır: $\frac{L(w+h) - L(w-h)}{2h}$ (*gradient checking*).
- ❌ **Sabit adım sayısında durmak.** Kayıptaki değişim çok küçüldüğünde (ör. $< 10^{-8}$) durmak veya doğrulama hatası artmaya başlayınca **erken durdurmak** (Ders 5) daha sağlıklıdır.

## 7. Mini Alıştırma

**Soru:** $L(w) = w^2$ fonksiyonunu $w_0 = 1$ noktasından başlayarak gradyan inişiyle küçültüyoruz ($L'(w) = 2w$).

1. $\eta = 0.25$ için ilk üç adımda $w$ değerleri nedir?
2. $\eta = 1$ ve $\eta = 1.5$ için ne olur?
3. Bu fonksiyon için kararlılık sınırı nedir?

<details>
<summary>Çözüm</summary>

Güncelleme: $w_{t+1} = w_t - \eta \cdot 2 w_t = (1 - 2\eta)\, w_t$. Yani her adımda $w$, $(1 - 2\eta)$ ile çarpılır.

1. $\eta = 0.25$ → çarpan $0.5$: $w_1 = 0.5,\; w_2 = 0.25,\; w_3 = 0.125$ → yakınsar.
2. $\eta = 1$ → çarpan $-1$: $w$ değeri $1, -1, 1, -1, \dots$ diye sonsuza dek salınır (ne yakınsar ne patlar).
   $\eta = 1.5$ → çarpan $-2$: $1, -2, 4, -8, \dots$ → **ıraksar**.
3. Yakınsama için $|1 - 2\eta| < 1$ olmalı → $0 < \eta < 1$. Bu, genel kuralla aynıdır: $L'' = \lambda = 2$ ve $\eta < 2/\lambda = 1$.
</details>

---
*Özet:* Gradyan inişi, gradyanın tersi yönünde $\eta$ büyüklüğünde adımlar atar. $\eta$ çok küçükse yavaş, $2/\lambda_{\max}$'tan büyükse ıraksar. Kayıp eğrisini izle, $\eta$'yı logaritmik ızgarada ara ve özellikleri ölçekle.
