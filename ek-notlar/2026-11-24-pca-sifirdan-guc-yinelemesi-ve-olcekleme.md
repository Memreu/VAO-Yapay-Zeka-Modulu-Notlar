# PCA'yı Sıfırdan Yazmak: Kovaryans Matrisi, Güç Yinelemesi ve Ölçekleme Tuzağı

> **Bağlantılı olduğu dersler:** Ders 6 – Boyut İndirgeme ve Temel Bileşenler Analizi (PCA), PCA + SVM entegrasyonu · Ders 3 – Keşifsel Veri Analizi (EDA)
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Hatırlatma: PCA Ne Arar?

Ders 6'da PCA'yı "verinin en çok yayıldığı yönleri bulup, veriyi bu yönlere izdüşürmek" diye özetledik. Bu yönlere **temel bileşen** (*principal component*, PC) denir:

- **PC1**, izdüşüm sonrası varyansı en büyük yapan birim vektördür.
- **PC2**, PC1'e dik olanlar arasında varyansı en büyük yapan yöndür. Diğerleri de aynı mantıkla devam eder.

Matematiksel olarak bu yönler, ortalaması çıkarılmış verinin **kovaryans matrisinin özvektörleridir**:

$$
C = \frac{1}{m-1} X_c^\top X_c, \qquad C\,v_k = \lambda_k\, v_k, \qquad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0
$$

| Sembol | Anlamı |
|---|---|
| $X_c$ | Her sütunundan ortalaması çıkarılmış $m \times d$ veri matrisi |
| $v_k$ | $k$. temel bileşenin yönü (birim uzunlukta) |
| $\lambda_k$ | O yöndeki varyans (özdeğer) |
| $\sum_k \lambda_k = \text{iz}(C)$ | Toplam varyans, yani özelliklerin varyanslarının toplamı |

**Açıklanan varyans oranı** $\lambda_k / \sum_j \lambda_j$'dir. "İlk 2 bileşen verinin %95'ini açıklıyor" gibi cümleler bu orandan gelir.

Kütüphanede tek satırda (`PCA(n_components=2).fit(X)`) bulunan bu işlemi bu notta kendimiz yazacağız. Böylece iki şeyi sayılarla göreceğiz:

1. Özvektörlerin **güç yinelemesi** (*power iteration*) ile nasıl bulunabildiğini,
2. Ölçekleme yapılmazsa PCA'nın neden **birimi en büyük olan sütunu** "en önemli bileşen" ilan ettiğini.

## 2. Güç Yinelemesi: En Büyük Özvektörü Bulmanın En Basit Yolu

Fikir şudur: Rastgele bir $v$ vektörünü $C$ ile tekrar tekrar çarpıp her adımda boyunu 1'e indirirsek, $v$ en büyük özdeğere ait özvektöre döner.

$$
v_{t+1} = \frac{C\,v_t}{\lVert C\,v_t \rVert}
$$

**Neden çalışır?** $v_0$'ı özvektörler cinsinden yazalım: $v_0 = a_1 v_1 + a_2 v_2 + \dots$. Her çarpım, $k$. bileşeni $\lambda_k$ katına çıkarır:

$$
C^t v_0 = \lambda_1^t \Big( a_1 v_1 + a_2 \big(\tfrac{\lambda_2}{\lambda_1}\big)^t v_2 + \dots \Big)
$$

$\lambda_2 / \lambda_1 < 1$ olduğundan diğer terimler üstel olarak söner ve geriye $v_1$ yönü kalır. Özdeğer ise **Rayleigh bölümü** ile okunur: $\lambda_1 \approx v^\top C v$.

**Sonraki bileşenler için deflasyon:** $v_1$ bulunduktan sonra onun katkısını matristen çıkarırız. Böylece yeni matrisin en büyük özvektörü $v_2$ olur:

$$
C' = C - \lambda_1 v_1 v_1^\top
$$

### Elle bir örnek

$C = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$, $v_0 = (1, 0)$. Gerçek cevap $\lambda_1 = 3$, $v_1 = (1,1)/\sqrt{2} \approx (0.7071,\, 0.7071)$.

| Adım | $C v_{t-1}$ | $v_t$ (normalize) | $v_t^\top C v_t$ |
|---|---|---|---|
| 1 | (2, 1) | (0.8944, 0.4472) | 2.8000 |
| 2 | (2.2361, 1.7889) | (0.7809, 0.6247) | 2.9756 |
| 3 | (2.1864, 2.0303) | (0.7328, 0.6805) | 2.9973 |
| 4 | (2.1460, 2.0937) | (0.7158, 0.6983) | 2.9997 |

Her adımda hata yaklaşık $\lambda_2/\lambda_1 = 1/3$ oranında küçülür. Özdeğerler birbirine ne kadar yakınsa yakınsama o kadar yavaşlar.

## 3. Çalışan Örnek: Boy, Kilo ve Gelir

Üç özellikli yapay bir veri seti kuralım:

- **boy** (cm): ortalama 170, standart sapma 10
- **kilo** (kg): boyla güçlü ilişkili ($r \approx 0.87$)
- **gelir** (TL): ortalama 30.000, standart sapma 8.000, diğerlerinden **bağımsız**

Sezgimiz şunu söyler: Boy ve kilo aynı bilgiyi büyük ölçüde tekrar ediyor ("vücut büyüklüğü"). Bu yüzden iyi bir PCA, bu ikisini tek bir bileşende birleştirmeli.

```python
import math
import random

random.seed(42)

# --- 1) Veri: boy (cm), kilo (kg), aylık gelir (TL) ---
n = 200
X = []
for _ in range(n):
    boy = random.gauss(170, 10)
    kilo = 70 + 0.9 * (boy - 170) + random.gauss(0, 5)   # boyla ilişkili
    gelir = random.gauss(30000, 8000)                     # bağımsız
    X.append([boy, kilo, gelir])
isimler = ["boy", "kilo", "gelir"]
d = len(X[0])

# --- 2) Yardımcı fonksiyonlar ---
def ortala(X):
    mu = [sum(s[j] for s in X) / len(X) for j in range(d)]
    return [[s[j] - mu[j] for j in range(d)] for s in X], mu

def standartlastir(X):
    Xc, mu = ortala(X)
    sd = [math.sqrt(sum(s[j] ** 2 for s in Xc) / (len(X) - 1)) for j in range(d)]
    return [[s[j] / sd[j] for j in range(d)] for s in Xc]

def kovaryans(Xc):
    m = len(Xc)
    return [[sum(s[i] * s[j] for s in Xc) / (m - 1) for j in range(d)] for i in range(d)]

def matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def guc_yinelemesi(A, iter_sayisi=1000, tol=1e-12):
    """En büyük özdeğeri ve özvektörü bulur (A simetrik, pozitif yarı tanımlı)."""
    v = [1.0] * len(A)
    for _ in range(iter_sayisi):
        w = matvec(A, v)
        w_norm = norm(w)
        w = [x / w_norm for x in w]
        if norm([a - b for a, b in zip(w, v)]) < tol:
            break
        v = w
    lam = sum(a * b for a, b in zip(v, matvec(A, v)))   # Rayleigh bölümü
    if max(v, key=abs) < 0:                              # işaret belirsizliği: sabitle
        v = [-x for x in v]
    return lam, v

def pca(Xc, k):
    A = kovaryans(Xc)
    iz = sum(A[i][i] for i in range(d))                  # toplam varyans
    bilesenler, ozdegerler = [], []
    for _ in range(k):
        lam, v = guc_yinelemesi(A)
        bilesenler.append(v)
        ozdegerler.append(lam)
        # Deflasyon: bulunan yönü matristen çıkar
        A = [[A[i][j] - lam * v[i] * v[j] for j in range(d)] for i in range(d)]
    return bilesenler, ozdegerler, iz

def rapor(baslik, Xc):
    bil, oz, iz = pca(Xc, d)
    print(f"\n=== {baslik} ===")
    print(f"Toplam varyans (iz): {iz:,.2f}")
    for i, (v, lam) in enumerate(zip(bil, oz), 1):
        yon = ", ".join(f"{isim}={x:+.3f}" for isim, x in zip(isimler, v))
        print(f"PC{i}: özdeğer={lam:12,.2f}  açıklanan oran=%{100*lam/iz:5.1f}  yön: {yon}")
    return bil, oz, iz

# --- 3) Ölçeklemeden PCA ---
Xc, mu = ortala(X)
rapor("Sadece ortalama çıkarıldı (ölçekleme YOK)", Xc)

# --- 4) Standartlaştırılmış PCA ---
Z = standartlastir(X)
bil, oz, iz = rapor("Standartlaştırıldı (z-skoru)", Z)

# --- 5) 3 boyuttan 2 boyuta indirgeme ve geri kurma ---
def izdusum(Z, bil, k):
    return [[sum(s[j] * bil[c][j] for j in range(d)) for c in range(k)] for s in Z]

def geri_kur(T, bil, k):
    return [[sum(t[c] * bil[c][j] for c in range(k)) for j in range(d)] for t in T]

for k in (1, 2, 3):
    T = izdusum(Z, bil, k)
    Zh = geri_kur(T, bil, k)
    hata = sum((a - b) ** 2 for s, sh in zip(Z, Zh) for a, b in zip(s, sh)) / (len(Z) - 1)
    print(f"k={k}: geri kurma hatası={hata:.4f}  atılan özdeğerlerin toplamı={sum(oz[k:]):.4f}")

# --- 6) Doğrulama: PC1 skorlarının varyansı = 1. özdeğer, PC'ler ilişkisiz ---
T = izdusum(Z, bil, 2)
var1 = sum(t[0] ** 2 for t in T) / (len(T) - 1)
kov12 = sum(t[0] * t[1] for t in T) / (len(T) - 1)
print(f"\nPC1 skor varyansı={var1:.4f} (özdeğer {oz[0]:.4f}),  kov(PC1, PC2)={kov12:.2e}")
```

**Çıktı:**

```text
=== Sadece ortalama çıkarıldı (ölçekleme YOK) ===
Toplam varyans (iz): 69,571,249.63
PC1: özdeğer=69,571,059.26  açıklanan oran=%100.0  yön: boy=+0.000, kilo=+0.000, gelir=+1.000
PC2: özdeğer=      177.23  açıklanan oran=%  0.0  yön: boy=+0.700, kilo=+0.714, gelir=-0.000
PC3: özdeğer=       13.14  açıklanan oran=%  0.0  yön: boy=+0.714, kilo=-0.700, gelir=+0.000

=== Standartlaştırıldı (z-skoru) ===
Toplam varyans (iz): 3.00
PC1: özdeğer=        1.87  açıklanan oran=% 62.3  yön: boy=+0.704, kilo=+0.704, gelir=+0.091
PC2: özdeğer=        0.99  açıklanan oran=% 33.1  yön: boy=-0.068, kilo=-0.061, gelir=+0.996
PC3: özdeğer=        0.14  açıklanan oran=%  4.6  yön: boy=-0.707, kilo=+0.707, gelir=-0.005
k=1: geri kurma hatası=1.1305  atılan özdeğerlerin toplamı=1.1305
k=2: geri kurma hatası=0.1377  atılan özdeğerlerin toplamı=0.1377
k=3: geri kurma hatası=0.0000  atılan özdeğerlerin toplamı=0.0000

PC1 skor varyansı=1.8695 (özdeğer 1.8695),  kov(PC1, PC2)=-4.17e-12
```

## 4. Sonuçları Okumak

### 4.1 Ölçekleme yokken: "PCA = gelir sütunu"

Gelirin varyansı yaklaşık $8000^2 = 6.4 \times 10^7$, boyunki ise yalnızca $10^2 = 100$. Kovaryans matrisinin izinin %99.9997'si gelirden gelir. Bu yüzden PC1 tam olarak gelir eksenidir ve "verinin %100'ünü açıklar".

Oysa gelir burada **en az bilgi taşıyan** özelliktir, çünkü diğer hiçbir şeyle ilişkili değildir. PCA'nın "varyans" dediği şey birimlere bağlıdır. Geliri bin TL cinsinden yazsaydık sonuç tamamen değişirdi. **Birimi değiştirince değişen bir analiz, verinin yapısını değil, ölçü biriminin seçimini anlatır.**

### 4.2 Standartlaştırınca: yapı ortaya çıkıyor

Her sütunu z-skoruna çevirince (ortalama 0, standart sapma 1) kovaryans matrisi **korelasyon matrisine** dönüşür ve toplam varyans $d = 3$ olur.

- **PC1 ≈ (boy + kilo)/√2:** "Vücut büyüklüğü" bileşeni. Özdeğeri $\approx 1 + r = 1.87$'dir ve tek başına verinin %62'sini açıklar.
- **PC2 ≈ gelir:** Bağımsız bir özellik kendi başına bir bileşen olur. Özdeğeri yaklaşık 1'dir, yani "bir sütun değerinde" bilgi taşır.
- **PC3 ≈ (kilo − boy)/√2:** "Boyuna göre kilo" farkı. Özdeğeri $1 - r \approx 0.14$, yani bu yön neredeyse gürültüdür ve atılabilir.

Yani 3 boyutlu veriyi 2 boyuta indirirken yalnızca %4.6 varyans kaybederiz.

### 4.3 Geri kurma hatası = atılan özdeğerlerin toplamı

Çıktıdaki `k=1` ve `k=2` satırları PCA'nın ikinci tanımını doğrular: PCA, **$k$ boyuta izdüşürüp geri döndürdüğümüzde ortalama kare hatayı en küçük yapan** doğrusal yöntemdir ve bu hata tam olarak atılan özdeğerlerin toplamına eşittir:

$$
\frac{1}{m-1}\sum_i \lVert z_i - \hat z_i \rVert^2 = \sum_{j > k} \lambda_j
$$

"En çok varyansı tut" ve "en az bilgi kaybet" aynı problemin iki yüzüdür.

### 4.4 Son satırdaki iki doğrulama

- PC1 skorlarının varyansı özdeğere eşittir (1.8695). Özdeğer, gerçekten o yöndeki varyanstır.
- PC1 ve PC2 skorları arasındaki kovaryans $\approx 10^{-12}$'dir, yani sayısal olarak sıfırdır. **Temel bileşenler birbiriyle ilişkisizdir.** Ders 6'daki PCA + SVM örneğinde PCA'nın çoklu bağlantıyı (*multicollinearity*) gidermesi bu özelliğe dayanır.

## 5. scikit-learn Karşılığı

Gerçek projede güç yinelemesi yerine kütüphane kullanılır. scikit-learn, kovaryans matrisini hiç kurmadan veriye doğrudan **tekil değer ayrışımı** (SVD) uygular. Bu yöntem sayısal olarak daha kararlıdır:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))  # %95 varyansı koruyacak kadar bileşen
T = pipe.fit_transform(X_train)
pca = pipe.named_steps["pca"]
print(pca.n_components_, pca.explained_variance_ratio_, pca.components_)
```

Bizim veride bu kod 2 bileşen seçer ($62.3 + 33.1 = 95.4\%$). Bileşen yönlerinin **işareti** bizim sonuçtan farklı çıkabilir. Bu bir hata değildir (bkz. sık hatalar).

## 6. Sık Yapılan Hatalar

- ❌ **Farklı birimdeki özelliklerde ölçeklemeden PCA yapmak.** Yukarıda gördüğümüz gibi en büyük birimli sütun "kazanır". Özellikler aynı birimdeyse ve büyüklükleri anlamlıysa (ör. aynı görüntünün piksel değerleri) ölçeklemeden yapmak doğru olabilir. Diğer bütün durumlarda önce `StandardScaler` uygulayın.
- ❌ **Ortalamayı çıkarmayı unutmak.** Ortalanmamış veride PC1 genellikle "orijinden veri bulutunun merkezine giden" yönü gösterir. Bu, yayılımla ilgili değil, ortalamayla ilgili bir yöndür. (`TruncatedSVD` bilerek ortalama çıkarmaz. Seyrek metin matrisleri için tasarlanmıştır, PCA ile karıştırmayın.)
- ❌ **Scaler'ı ve PCA'yı tüm veriye fit edip sonra bölmek.** Ortalama, standart sapma ve bileşen yönleri test verisini de "görmüş" olur. Bu bir [veri sızıntısıdır](2026-11-18-veri-sizintisi-data-leakage.md). İkisini de bir `Pipeline` içinde, yalnızca eğitim verisine fit edin.
- ❌ **Bileşenlerin işaretini yorumlamak.** $v$ bir özvektörse $-v$ de özvektördür. Farklı kütüphaneler, sürümler veya rastgele başlangıçlar işareti ters çevirebilir. Anlamlı olan yükler (*loadings*) arasındaki **göreli** işaretlerdir. Örneğin PC3'te boy ile kilonun zıt işaretli olması anlamlıdır.
- ❌ **"Yüksek varyanslı yön = hedefi en iyi ayıran yön" sanmak.** PCA denetimsizdir, etiketlere hiç bakmaz. Sınıfları ayıran bilgi küçük bir bileşende saklı olabilir. Etiket kullanan boyut indirgeme gerekiyorsa Ders 6'daki **LDA**'ya bakın ve bileşen sayısını modelin [çapraz doğrulama](2026-11-21-k-katli-capraz-dogrulama-ile-model-secimi.md) skoruna göre seçin.
- ❌ **Güç yinelemesine sıfır vektörüyle veya en büyük özvektöre dik bir vektörle başlamak.** $a_1 = 0$ ise yöntem $v_1$'i asla bulamaz. Pratikte rastgele başlangıç bu riski ortadan kaldırır. Ayrıca $\lambda_1 \approx \lambda_2$ ise yakınsama çok yavaşlar. Deflasyon da hataları biriktirdiği için çok sayıda bileşen gerekiyorsa SVD tercih edilir.

## 7. Mini Alıştırma

**Soru:** İki özellik standartlaştırılmış ve aralarındaki korelasyon $r = 0.8$. Korelasyon matrisi:

$$
R = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}
$$

1. $v = (1, 1)/\sqrt{2}$ ve $u = (1, -1)/\sqrt{2}$ vektörlerinin $R$'nin özvektörleri olduğunu gösterin ve özdeğerleri bulun.
2. PC1'in açıklanan varyans oranı nedir? %95 varyansı korumak için kaç bileşen gerekir?
3. Korelasyon $r = 0$ olsaydı PCA ne yapardı? Güç yinelemesi bu durumda neden işe yaramaz?

<details>
<summary>Çözüm</summary>

1. $R v = \tfrac{1}{\sqrt 2}(1 + 0.8,\; 0.8 + 1) = 1.8\, v$ olduğundan $\lambda_1 = 1.8$ olur.
   $R u = \tfrac{1}{\sqrt 2}(1 - 0.8,\; 0.8 - 1) = 0.2\, u$ olduğundan $\lambda_2 = 0.2$ olur.
   Genel olarak $2 \times 2$ korelasyon matrisinin özdeğerleri $1 \pm r$'dir. (Koddaki boy–kilo çifti için $1 + 0.87 \approx 1.87$ çıkması bu yüzdendir.)
2. Oran $1.8 / (1.8 + 0.2) = 0.90$, yani %90'dır. %95 için **2 bileşen** gerekir, ki bu da hiç boyut indirgememek demektir. Eşiği %90'a çekersek tek bileşen yeter.
3. $r = 0$ iken $R = I$ (birim matris) olur ve iki özdeğer de 1'dir. Bütün yönler eşit varyanslıdır, yani **hiçbir yön diğerinden üstün değildir**. Herhangi bir dik eksen çifti "temel bileşen" sayılabilir, bu yüzden boyut indirgemenin bir anlamı kalmaz. Güç yinelemesinde $I v = v$ olduğundan vektör hiç değişmez ve başlangıçta hangi yöndeyse orada kalır. Yakınsama oranı $\lambda_2/\lambda_1 = 1$ olduğundan "kazanan" bir yön çıkmaz.
</details>

---
*Özet:* PCA, kovaryans matrisinin özvektörlerini bulmaktır. Güç yinelemesi bunu "matrisle tekrar tekrar çarp ve normalize et" kadar basit bir döngüyle yapar ve deflasyonla sonraki bileşenlere geçer. Özdeğerler her yöndeki varyansı verir. Atılan özdeğerlerin toplamı da geri kurma hatasına tam olarak eşittir. Ancak "varyans" birime bağlıdır: Farklı birimlerdeki özellikleri **standartlaştırmadan** PCA yapmak, verinin yapısını değil en büyük birimli sütunu bulur.
