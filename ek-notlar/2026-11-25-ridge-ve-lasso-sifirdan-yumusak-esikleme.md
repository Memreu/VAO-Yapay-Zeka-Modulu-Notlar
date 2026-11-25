# Ridge ve Lasso'yu Sıfırdan Yazmak: Katsayılar Neden Küçülür, Neden Tam Sıfır Olur?

> **Bağlantılı olduğu dersler:** Ders 5 – Bölüm 3: Regresyonda Düzenlileştirme (Regularization) ve Aşırı Uyum Yönetimi · Ders 4 – Aşırı uyum ve eksik uyum
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Hatırlatma: İki Ceza, İki Farklı Davranış

Ders 5'te doğrusal regresyonun kayıp fonksiyonuna bir ceza terimi ekleyerek aşırı uyumu frenlediğimizi gördük:

$$
J_{\text{Ridge}}(b) = \lVert y - Xb \rVert^2 + \lambda \sum_j b_j^2
\qquad\qquad
J_{\text{Lasso}}(b) = \frac{1}{2m}\lVert y - Xb \rVert^2 + \alpha \sum_j |b_j|
$$

Derste iki önemli cümle vardı:

- Ridge ($L^2$) katsayıları **orantılı olarak küçültür** ama hiçbirini tam sıfır yapmaz.
- Lasso ($L^1$) bazı katsayıları **tam olarak sıfıra** eşitler, yani otomatik öznitelik seçimi yapar. Kapalı form çözümü yoktur, sayısal olarak çözülür.

Bu notta iki modeli de kütüphanesiz yazıp bu cümleleri sayılarla göreceğiz. Ayrıca "tam sıfır" davranışının nereden geldiğini, tek bir formülle (**yumuşak eşikleme**) açıklayacağız.

> **Ölçek notu:** Yukarıdaki iki formül, scikit-learn'deki `Ridge(alpha=λ)` ve `Lasso(alpha=α)` sınıflarının kullandığı ölçeklerle aynıdır. Lasso'daki $\frac{1}{2m}$ çarpanı yüzünden iki modelin "alpha" değerleri doğrudan karşılaştırılamaz. Ders 5'teki $\frac{1}{N}$'li formül de aynı fikri farklı bir ölçekle yazar.

## 2. Tek Boyutta Sezgi: Bölmek ve Çıkarmak

Tek bir standartlaştırılmış özellik ($\frac{1}{m}\sum z_i^2 = 1$) ve ortalaması çıkarılmış $y$ olsun. En küçük kareler (EKK) çözümü $\hat b = \frac{1}{m}\sum z_i y_i$'dir. Türev alıp sıfıra eşitlersek:

| Model | Çözüm | Etkisi |
|---|---|---|
| Ridge | $b = \hat b \cdot \dfrac{m}{m + \lambda}$ | Her katsayıyı **aynı oranda** böler. $\hat b \ne 0$ ise sonuç asla 0 olmaz. |
| Lasso | $b = S(\hat b, \alpha) = \text{işaret}(\hat b)\cdot\max(|\hat b| - \alpha,\ 0)$ | Her katsayıdan **sabit bir miktar** çıkarır. $|\hat b| \le \alpha$ ise sonuç **tam 0** olur. |

$S$ fonksiyonuna **yumuşak eşikleme** (*soft thresholding*) denir. Lasso'nun sıfır üretmesinin sebebi budur: $|b|$ fonksiyonunun sıfırdaki "köşesi" nedeniyle, küçük bir EKK katsayısını sıfırdan uzaklaştırmanın cezası, kayıptaki kazançtan her zaman büyüktür.

**Çok boyutta koordinat inişi:** Lasso'yu her seferinde **tek bir katsayıyı** güncelleyerek çözeriz. Diğer katsayılar sabitken $b_j$ için problem yukarıdaki tek boyutlu probleme dönüşür:

$$
\rho_j = \frac{1}{m}\sum_i z_{ij}\,\big(r_i + z_{ij} b_j\big), \qquad b_j \leftarrow S(\rho_j,\ \alpha)
$$

Burada $r = y - Zb$ güncel artıktır (*residual*). $\rho_j$, "$b_j$ dışındaki her şey sabitken $j$. özelliğin artıkla ilişkisi" demektir. Bütün katsayılar değişmeyi bırakana kadar turlar tekrarlanır. scikit-learn'deki `Lasso` da bu yöntemi kullanır.

## 3. Çalışan Örnek: 15 Örnek, 8 Özellik, 3 Gerçek Etki

Senaryo: $y = 3x_0 - 2x_1 + 1.5x_2 + \text{gürültü}$. Diğer 5 özelliğin etkisi yoktur. Bunlardan $x_3$, $x_0$'ın "ikizidir" ($r \approx 0.9$). Yani $x_0$ ile birlikte hareket eder ama $y$'yi kendisi etkilemez. Eğitim için yalnızca **15 örnek** var. Bu, 8 katsayı için çok az veri demektir ve EKK'nın aşırı uyması beklenir. Test için 2000 örnek üretiyoruz.

```python
import random, math

# --- 1) Veri: 8 özellik, yalnızca ilk 3'ü gerçekten etkili ---
random.seed(2)
GERCEK = [3.0, -2.0, 1.5, 0, 0, 0, 0, 0]
d = len(GERCEK)

def veri_uret(m):
    X, y = [], []
    for _ in range(m):
        x = [random.gauss(0, 1) for _ in range(d)]
        x[3] = 0.9 * x[0] + 0.45 * random.gauss(0, 1)   # x3, x0'ın "ikizi" (r ≈ 0.9) ama etkisiz
        X.append(x)
        y.append(sum(b * xi for b, xi in zip(GERCEK, x)) + random.gauss(0, 2.0))
    return X, y

X_tr, y_tr = veri_uret(15)    # az veri: 15 örnek, 8 özellik
X_te, y_te = veri_uret(2000)

# --- 2) Standartlaştırma (yalnızca eğitim verisine fit) ---
m = len(X_tr)
ort = [sum(r[j] for r in X_tr) / m for j in range(d)]
ss  = [math.sqrt(sum((r[j] - ort[j]) ** 2 for r in X_tr) / m) for j in range(d)]
y_ort = sum(y_tr) / m
olcekle = lambda X: [[(r[j] - ort[j]) / ss[j] for j in range(d)] for r in X]
Z_tr, Z_te = olcekle(X_tr), olcekle(X_te)
yc = [v - y_ort for v in y_tr]

def tahmin(Z, b): return [y_ort + sum(bj * zj for bj, zj in zip(b, z)) for z in Z]
def mse(y, yh):   return sum((a - c) ** 2 for a, c in zip(y, yh)) / len(y)

# --- 3) Ridge: kapalı form (ZᵀZ + λI) b = Zᵀy, Gauss eliminasyonu ile ---
def coz(A, c):
    n = len(c); M = [A[i][:] + [c[i]] for i in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k])); M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1): M[i][j] -= f * M[k][j]
    b = [0.0] * n
    for i in reversed(range(n)):
        b[i] = (M[i][n] - sum(M[i][j] * b[j] for j in range(i + 1, n))) / M[i][i]
    return b

ZtZ = [[sum(z[i] * z[j] for z in Z_tr) for j in range(d)] for i in range(d)]
Zty = [sum(z[i] * v for z, v in zip(Z_tr, yc)) for i in range(d)]
def ridge(lam):
    return coz([[ZtZ[i][j] + (lam if i == j else 0) for j in range(d)] for i in range(d)], Zty)

# --- 4) Lasso: koordinat inişi + yumuşak eşikleme ---
#     amaç: (1/2m)·||y - Zb||² + α·||b||₁   (sklearn Lasso ile aynı ölçek)
def yumusak_esik(r, a): return math.copysign(max(abs(r) - a, 0.0), r)
def lasso(alpha, tur=1000, tol=1e-10):
    b = [0.0] * d; art = yc[:]                       # artık = y - Zb
    for _ in range(tur):
        en_buyuk = 0.0
        for j in range(d):
            zj = [z[j] for z in Z_tr]
            rho = sum(a * zz for a, zz in zip(art, zj)) / m + b[j]   # (standart: (1/m)Σz² = 1)
            yeni = yumusak_esik(rho, alpha)
            if yeni != b[j]:
                art = [a - (yeni - b[j]) * zz for a, zz in zip(art, zj)]
                en_buyuk = max(en_buyuk, abs(yeni - b[j])); b[j] = yeni
        if en_buyuk < tol: break
    return b

fmt = lambda b: " ".join(f"{v + 0.0:6.2f}" for v in b)   # +0.0: "-0.00" yazmasın
print("Gerçek katsayılar:", GERCEK, "| eğitim std:", " ".join(f"{s:.2f}" for s in ss))
print("\nRIDGE      λ  |" + "".join(f"    b{j}" for j in range(d)) + " | test MSE")
for lam in [0, 1, 3, 10, 30, 100]:
    b = ridge(lam); print(f"       {lam:6g}  | {fmt(b)} | {mse(y_te, tahmin(Z_te, b)):6.2f}")

a_max = max(abs(sum(z[j] * v for z, v in zip(Z_tr, yc)) / m) for j in range(d))
print(f"\nα_max = max|zⱼᵀy|/m = {a_max:.3f}  (bu değerden itibaren bütün katsayılar 0)")
print("LASSO      α  |" + "".join(f"    b{j}" for j in range(d)) + " | test MSE | sıfır sayısı")
for alpha in [0.001, 0.1, 0.3, 0.6, 1.0, round(a_max, 3) + 0.001]:
    b = lasso(alpha)
    print(f"       {alpha:6g}  | {fmt(b)} | {mse(y_te, tahmin(Z_te, b)):6.2f}   | {sum(v == 0 for v in b)}")

# --- 5) Doğrulama: Lasso KKT koşulları (α = 0.3) ---
alpha = 0.3; b = lasso(alpha)
art = [v - sum(bj * zj for bj, zj in zip(b, z)) for z, v in zip(Z_tr, yc)]
g = [sum(z[j] * a for z, a in zip(Z_tr, art)) / m for j in range(d)]
ok = all((abs(g[j] - alpha * math.copysign(1, b[j])) < 1e-6) if b[j] != 0 else (abs(g[j]) <= alpha + 1e-9) for j in range(d))
print("\nKKT (α=0.3): |zⱼᵀr/m| =", fmt(g), "→ koşullar sağlanıyor mu?", ok)
```

Çıktı:

```text
Gerçek katsayılar: [3.0, -2.0, 1.5, 0, 0, 0, 0, 0] | eğitim std: 1.08 0.86 1.24 1.03 0.93 1.17 0.79 0.89

RIDGE      λ  |    b0    b1    b2    b3    b4    b5    b6    b7 | test MSE
            0  |   3.00  -2.31   2.01  -0.25   0.35  -0.09  -0.50   1.26 |   7.02
            1  |   2.24  -1.96   1.80   0.54   0.30  -0.20  -0.56   1.13 |   6.36
            3  |   1.78  -1.56   1.60   0.98   0.30  -0.28  -0.59   0.92 |   6.09
           10  |   1.34  -1.00   1.22   1.12   0.35  -0.31  -0.54   0.61 |   6.81
           30  |   0.91  -0.54   0.78   0.87   0.32  -0.25  -0.37   0.36 |   9.19
          100  |   0.44  -0.21   0.36   0.44   0.18  -0.13  -0.17   0.17 |  13.46

α_max = max|zⱼᵀy|/m = 4.117  (bu değerden itibaren bütün katsayılar 0)
LASSO      α  |    b0    b1    b2    b3    b4    b5    b6    b7 | test MSE | sıfır sayısı
        0.001  |   2.99  -2.31   2.01  -0.24   0.35  -0.09  -0.50   1.25 |   6.99   | 0
          0.1  |   2.99  -1.92   2.19   0.00   0.00   0.00  -0.29   0.75 |   4.97   | 3
          0.3  |   3.06  -1.40   2.17   0.00   0.00   0.00  -0.11   0.19 |   4.35   | 3
          0.6  |   2.86  -1.00   1.96   0.07   0.00   0.00   0.00   0.00 |   4.90   | 4
            1  |   2.23  -0.53   1.52   0.48   0.00   0.00   0.00   0.00 |   6.38   | 4
        4.118  |   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00 |  19.17   | 8

KKT (α=0.3): |zⱼᵀr/m| =   0.30  -0.30   0.30   0.27   0.04  -0.16  -0.30   0.30 → koşullar sağlanıyor mu? True
```

## 4. Sonuçları Okumak

Özellikler standartlaştırıldığı için gerçek katsayıların bu ölçekteki karşılığı yaklaşık "gerçek katsayı × eğitim std" olur: $b_0 \approx 3.24$, $b_1 \approx -1.72$, $b_2 \approx 1.86$, geri kalanlar 0.

### 4.1 EKK (λ = 0) gürültüyü de öğreniyor

$\lambda = 0$ satırı düz en küçük karelerdir. Etkisiz özelliklere de belirgin katsayılar veriyor ($b_7 = 1.26$, $b_6 = -0.50$). 15 örnekte 8 katsayı tahmin ederken model, gürültüdeki rastlantısal desenleri de "açıklamaya" çalışıyor. Test MSE'si 7.02.

### 4.2 Ridge: her şeyi biraz küçültüyor, hiçbir şeyi silmiyor

- $\lambda$ büyüdükçe bütün katsayılar sıfıra yaklaşıyor ama **hiçbiri tam 0 olmuyor**. $\lambda = 100$'de bile 8 katsayının hepsi duruyor.
- Test MSE'si önce iyileşiyor ($\lambda = 3$'te 6.09), sonra kötüleşiyor ($\lambda = 100$'de 13.46). Bu, Ders 4'teki varyans–yanlılık dengesinin ta kendisi: Küçük $\lambda$ varyansı azaltır, büyük $\lambda$ modeli eksik uyuma (*underfitting*) iter.
- **İkiz etkisi:** $x_3$ etkisiz olduğu halde $b_3$, EKK'da $-0.25$ iken $\lambda = 10$'da $1.12$'ye **çıkıyor**. Ridge, birbirine benzeyen iki özellik arasında ağırlığı paylaştırmayı sever, çünkü $1.5^2 + 1.5^2 = 4.5 < 3^2 = 9$. Toplam etki aynı kalırken ceza yarıya iner. Bu yüzden Ridge katsayıları "hangi özellik önemli?" sorusuna dikkatli yorumlanmalıdır.

### 4.3 Lasso: küçültüyor ve seçiyor

- $\alpha = 0.3$'te etkisiz özelliklerden $x_3, x_4, x_5$ **tam olarak sıfır**. $b_6$ ve $b_7$ çok küçülmüş. Gerçek etkiler ise yerinde duruyor: $3.06,\ -1.40,\ 2.17$.
- Test MSE'si **4.35**. Bu, Ridge'in en iyi değerinden (6.09) belirgin şekilde daha iyi. Sebebi bu verinin **seyrek** olması: Gerçek modelde 8 katsayıdan yalnızca 3'ü sıfırdan farklı. Lasso'nun varsayımı veriye uyuyor.
- İkizler arasında Lasso **birini seçiyor**: $x_3$'ü sıfırlayıp ağırlığı $x_0$'a veriyor. Ancak $\alpha = 1$'de $b_0$ çok küçülünce $b_3$ yeniden 0.48'e çıkıyor. Ders 5'te belirtilen "ilişkili özelliklerden rastgele birini seçme" davranışının kararsızlığı budur ve **Elastic Net** bu yüzden vardır.
- **$\alpha_{\max}$:** $\alpha \ge \max_j |\tfrac{1}{m} z_j^\top y|$ olduğunda başlangıçtaki sıfır vektörü zaten optimumdur ve bütün katsayılar 0 kalır. Model sadece ortalamayı tahmin eder (test MSE 19.17). `LassoCV`, deneyeceği $\alpha$ ızgarasını bu değerden aşağıya doğru kurar.

### 4.4 Çözümü doğrulamak: KKT koşulları

Son satır, bulduğumuz çözümün gerçekten optimum olduğunu kontrol ediyor. Lasso'nun optimumunda her özellik için şu koşul sağlanmalıdır:

$$
\frac{1}{m} z_j^\top r =
\begin{cases}
\alpha \cdot \text{işaret}(b_j) & b_j \ne 0 \text{ ise} \\
\text{değeri } [-\alpha,\ \alpha] \text{ aralığında} & b_j = 0 \text{ ise}
\end{cases}
$$

Yorumu şöyle: Sıfırdan farklı her katsayı, artıkla ilişkisi **tam olarak $\alpha$** olana kadar büyür. Artıkla ilişkisi $\alpha$'yı geçemeyen özellik ise modele hiç giremez. Çıktıda sıfır olmayan katsayılar için değerler $\pm 0.30$, sıfır olanlar ($b_3, b_4, b_5$) için $0.27,\ 0.04,\ -0.16$, yani hepsi $\le 0.3$.

## 5. scikit-learn Karşılığı

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LassoCV

ridge = make_pipeline(StandardScaler(), Ridge(alpha=3)).fit(X_tr, y_tr)
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.3)).fit(X_tr, y_tr)
print(ridge[-1].coef_.round(2))   # [ 1.78 -1.56  1.6   0.98  0.3  -0.28 -0.59  0.92]
print(lasso[-1].coef_.round(2))   # [ 3.06 -1.4   2.17  0.    0.   -0.   -0.11  0.19]

# alpha'yı test verisine bakarak değil, çapraz doğrulamayla seçin:
cv = make_pipeline(StandardScaler(), LassoCV(cv=5)).fit(X_tr, y_tr)
print(cv[-1].alpha_)
```

Aynı veriyle (scikit-learn 1.9.1) iki modelin katsayıları ve test MSE'leri (6.09 ve 4.35) bizim kodla iki ondalığa kadar aynı çıkıyor. Bizim kodda $\sigma$ hesaplanırken $m$'ye bölünmesi `StandardScaler` ile uyum içindir.

## 6. Sık Yapılan Hatalar

- ❌ **Ölçeklemeden düzenlileştirme yapmak.** Ceza, katsayının büyüklüğüne bakar ve katsayının büyüklüğü özelliğin birimine bağlıdır. "Gelir (TL)" sütununun katsayısı doğal olarak küçük olduğu için neredeyse hiç cezalandırılmaz, "yaş (on yıl)" sütunu ise ağır cezalandırılır. Önce `StandardScaler` uygulayın. (Benzer bir birim tuzağı için bkz. [PCA notu](2026-11-24-pca-sifirdan-guc-yinelemesi-ve-olcekleme.md).)
- ❌ **Sabit terimi (intercept) cezalandırmak.** Sabit terim yalnızca $y$'nin ortalamasını taşır. Onu sıfıra çekmek modeli anlamsızca kaydırır. Yukarıdaki kodda $y$'yi ortalayıp sabit terimi cezanın dışında bırakıyoruz. scikit-learn de varsayılan olarak böyle yapar.
- ❌ **$\lambda$'yı test setine bakarak seçmek.** Tablolarda test MSE'sini **öğretmek için** gösterdik. Gerçek bir projede $\lambda$, yalnızca eğitim verisi üzerinde [k-katlı çapraz doğrulama](2026-11-21-k-katli-capraz-dogrulama-ile-model-secimi.md) ile seçilir. Aksi halde test skoru iyimser olur. Bu bir [veri sızıntısıdır](2026-11-18-veri-sizintisi-data-leakage.md).
- ❌ **Lasso'yu düz gradyan inişiyle çözüp "sıfır çıkmadı" demek.** $|b|$'nin sıfırda türevi yoktur. Alt-gradyan ile yapılan düz [gradyan inişi](2026-11-20-gradyan-inisi-ve-ogrenme-orani.md), katsayıları sıfırın etrafında küçük değerlerde salındırır ama **tam sıfır** üretmez. Koordinat inişi veya proksimal gradyan (yine yumuşak eşikleme kullanır) gerekir.
- ❌ **Lasso'nun seçtiği özellikleri "tek doğru özellikler" sanmak.** İlişkili özellikler arasında seçim kararsızdır. Veri biraz değişince başka bir ikiz seçilebilir (yukarıda $\alpha = 1$'de $x_3$'ün geri gelmesi gibi). Yorum yaparken farklı bootstrap örneklemlerinde seçimin ne kadar tutarlı olduğuna bakın veya Elastic Net kullanın.
- ❌ **scikit-learn'deki `C` ile `alpha`'yı karıştırmak.** `Ridge` ve `Lasso`'da `alpha` büyüdükçe ceza artar. `LogisticRegression` ve `SVC`'deki `C` ise **ters** düzenlileştirme gücüdür ($C \approx 1/\lambda$): `C` küçüldükçe ceza artar. Ders 5'teki el yazısı rakam örneğinde `C=0.1` güçlü bir cezadır.

## 7. Mini Alıştırma

**Soru:** Standartlaştırılmış tek bir özellik için $m = 20$ örnekle iki farklı veri setinde EKK katsayıları $\hat b_A = 0.5$ ve $\hat b_B = 2.0$ bulunmuş olsun (Bölüm 2'deki formüller geçerli).

1. Ridge ($\lambda = 20$) ile iki katsayı ne olur?
2. Lasso ($\alpha = 0.8$) ile iki katsayı ne olur?
3. Hangi model katsayıların **oranını** ($\hat b_B / \hat b_A = 4$) korur? Bu, "önemsiz özelliği silme" açısından ne anlama gelir?

<details>
<summary>Çözüm</summary>

1. Çarpan $\frac{m}{m+\lambda} = \frac{20}{40} = 0.5$ olur. $b_A = 0.25$, $b_B = 1.0$. İkisi de yarıya inmiştir.
2. $S(0.5,\ 0.8) = \text{işaret}(0.5)\cdot\max(0.5 - 0.8,\ 0) = 0$ ve $S(2.0,\ 0.8) = 2.0 - 0.8 = 1.2$ olur. Yani $b_A = 0$, $b_B = 1.2$.
3. Ridge oranı korur ($1.0 / 0.25 = 4$), çünkü her katsayıyı aynı sayıya böler. Bu yüzden zayıf bir özelliği asla silmez, yalnızca küçültür. Lasso ise herkesten aynı miktarı ($\alpha$) çıkarır. Küçük katsayının üzerinde "çıkarılacak kadar" değer kalmadığı için o tamamen silinir, büyük katsayı ise görece az etkilenir ($2.0 \to 1.2$, %40 küçülme). Bunun bedeli, Lasso'nun büyük ve gerçek etkileri de **sabit bir miktar** küçültmesidir (yanlılık). Bu yüzden bazen Lasso yalnızca özellik seçmek için kullanılır ve seçilen özelliklerle ceza uygulanmadan yeniden bir EKK modeli kurulur (*relaxed lasso*).
</details>

---
*Özet:* Ridge katsayıları **bölerek** küçültür ve ilişkili özellikler arasında ağırlığı paylaştırır. Lasso **çıkararak** küçültür. Yumuşak eşikleme sayesinde artıkla ilişkisi $\alpha$'yı geçemeyen özellikleri tam olarak sıfırlar. Az veri ve çok sayıda etkisiz özellik olduğunda (seyrek gerçek model) Lasso hem daha iyi tahmin yapar hem de daha okunaklı bir model verir. Her iki yöntemde de önce **ölçeklemek**, sabit terimi cezalandırmamak ve $\lambda$/$\alpha$'yı **çapraz doğrulamayla** seçmek şarttır.
