# Dikkat (Attention) Mekanizmasını Sıfırdan Yazmak: Q·K, √d Ölçeklemesi ve Nedensel Maske

> **Bağlantılı olduğu dersler:** Ders 3 – Bölüm 6: Transformer Mimarisi (6.4 Softmax ve Sıcaklık) · Ders 3 – Bölüm 7: Attention Mekanizması (7.2 Query/Key/Value, 7.3 Maskeleme)
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Hatırlatma: Kelimeler Birbirine Nasıl "Bakar"?

Ders 3'te dikkat mekanizmasını üç vektörle anlatmıştık:

- **Query (Sorgu, $q$):** "Benim ne tür bir bilgiye ihtiyacım var?"
- **Key (Anahtar, $k$):** "Bende ne tür bir bilgi var?"
- **Value (Değer, $v$):** "Eşleşirsek sana aktaracağım bilgi bu."

Sorgu ile anahtarların nokta çarpımı bir uyum skoru verir, softmax bu skorları toplamı 1 olan ağırlıklara çevirir, ağırlıklarla Value'ların ortalaması alınır. Derste bunun bir de **maskeli** hali vardı: gelecekteki kelimelerin skorları softmax'ten önce $-\infty$ yapılır.

Bu notta bu cümlelerin hepsini kısa, kütüphanesiz bir kodla çalıştıracağız ve derste formülde geçip üzerinde durmadığımız bir ayrıntıyı, **$\sqrt{d}$'ye bölmeyi**, sayılarla açıklayacağız.

## 2. Tek Formül

Transformer makalesindeki (Vaswani vd., 2017) **ölçekli nokta çarpımı dikkati** (*scaled dot-product attention*):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

Satır satır okursak, $i$. tokenin çıktısı:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}, \qquad
w_{ij} = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}, \qquad
c_i = \sum_j w_{ij}\, v_j
$$

| Sembol | Anlamı | Boyut |
|---|---|---|
| $Q, K$ | Her tokenin sorgu ve anahtar vektörleri (satır satır) | $n \times d$ |
| $V$ | Her tokenin değer vektörleri | $n \times d_v$ |
| $QK^\top$ | Her sorgu–anahtar çiftinin uyum skoru | $n \times n$ |
| $W$ (softmax sonrası) | **Dikkat deseni**: her satırın toplamı 1 | $n \times n$ |
| $c_i$ | $i$. tokenin bağlamla zenginleşmiş yeni vektörü | $d_v$ |

Gerçek modellerde $Q = XW_Q$, $K = XW_K$, $V = XW_V$ şeklinde girdiden öğrenilen üç matrisle üretilir. Bu notta mekanizmaya odaklanmak için $Q$, $K$, $V$'yi doğrudan elle veriyoruz.

> **Önemli:** Softmax **her satırda ayrı ayrı** (yani her sorgu için, anahtarlar üzerinden) uygulanır. "Bu token, diğer tokenlere dikkatini nasıl paylaştırıyor?" sorusunun cevabı bir satırdır.

## 3. Elle Bir Satır: "banka" Neye Bakıyor?

Üç token, $d = 2$:

| Token | $q$ | $k$ | $v$ |
|---|---|---|---|
| banka | $[1, 0]$ | $[2, 0]$ | $[1, 0]$ |
| nehir | $[0, 1]$ | $[0, 2]$ | $[0, 1]$ |
| kenarı | $[1, 1]$ | $[1, 1]$ | $[0.5, 0.5]$ |

"banka" sorgusunun skorları ($\sqrt{2} \approx 1.414$):

$$
s = \left[\tfrac{1\cdot2 + 0\cdot0}{1.414},\ \tfrac{0}{1.414},\ \tfrac{1\cdot1+0\cdot1}{1.414}\right] = [1.414,\ 0,\ 0.707]
$$

$$
e^{s} = [4.113,\ 1,\ 2.028], \quad \text{toplam} = 7.141 \quad\Rightarrow\quad w = [0.576,\ 0.140,\ 0.284]
$$

$$
c_{\text{banka}} = 0.576\,[1,0] + 0.140\,[0,1] + 0.284\,[0.5,0.5] = [0.718,\ 0.282]
$$

"banka" kendi vektörünü büyük ölçüde korudu ama "nehir"den ve "kenarı"ndan da biraz bilgi çekti. Ders 3'teki "yüz" örneğinde olduğu gibi, kelimenin anlamı bağlamla **güncellendi**.

**Maskeli halde** "banka" ilk token olduğu için yalnızca kendisini görür: $w = [1, 0, 0]$, yani $c_{\text{banka}} = v_{\text{banka}}$. "nehir" ise yalnızca ilk iki skoru kullanır: $\text{softmax}([0, 1.414]) = [0.196, 0.804]$.

## 4. Çalışan Örnek: Dikkat, Ölçekleme ve Maske Testleri

Kod beş parçadan oluşur: (1) yapı taşları, (2) yukarıdaki elle örnek, (3) $\sqrt{d}$ deneyi, (4) sıra bilgisi testi, (5) maske sızıntı testi.

```python
import math, random

# --- 1) Yapı taşları ---
def softmax(skorlar):
    m = max(skorlar)                       # taşmayı önlemek için en büyüğü çıkar
    us = [math.exp(s - m) for s in skorlar]  # exp(-inf) = 0.0 olur
    t = sum(us)
    return [u / t for u in us]

def nokta(a, b):
    return sum(x * y for x, y in zip(a, b))

def dikkat(Q, K, V, maske=False):
    """Ölçekli nokta çarpımı dikkati: softmax(Q·Kᵀ / √d) · V"""
    d = len(K[0])
    cikti, agirliklar = [], []
    for i, q in enumerate(Q):
        skor = [nokta(q, k) / math.sqrt(d) for k in K]
        if maske:                          # nedensel maske: i. token j > i'yi göremez
            skor = [s if j <= i else -math.inf for j, s in enumerate(skor)]
        w = softmax(skor)
        agirliklar.append(w)
        cikti.append([sum(w[j] * V[j][c] for j in range(len(V)))
                      for c in range(len(V[0]))])
    return cikti, agirliklar

def yazdir(baslik, M, satir_adlari):
    print(baslik)
    for ad, satir in zip(satir_adlari, M):
        print(f"  {ad:>6} " + " ".join(f"{x:6.3f}" for x in satir))

# --- 2) Elle izlenebilir örnek: 3 token, d = 2 ---
tokenler = ["banka", "nehir", "kenarı"]
Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
V = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

C, W = dikkat(Q, K, V)
yazdir("Dikkat ağırlıkları (satır = sorgu, sütun = anahtar):", W, tokenler)
yazdir("Çıktılar (Value'ların ağırlıklı ortalaması):", C, tokenler)
print("Satır toplamları:", [round(sum(w), 6) for w in W])

C, W = dikkat(Q, K, V, maske=True)
yazdir("Nedensel maskeli ağırlıklar:", W, tokenler)

# --- 3) Neden √d'ye bölüyoruz? ---
random.seed(0)
def rastgele_vektor(d):
    return [random.gauss(0, 1) for _ in range(d)]

def entropi(p):
    return -sum(x * math.log(x) for x in p if x > 0)

print("\n   d | skor std (ölçeksiz) | max ağırlık ölçeksiz / ölçekli | entropi ölçeksiz / ölçekli")
n = 10   # anahtar sayısı; düzgün dağılımın entropisi ln(10) = 2.303
for d in [4, 16, 64, 256, 1024]:
    stdler, mx_h, mx_o, en_h, en_o = [], [], [], [], []
    for _ in range(200):
        q = rastgele_vektor(d)
        Ks = [rastgele_vektor(d) for _ in range(n)]
        s = [nokta(q, k) for k in Ks]
        ort = sum(s) / n
        stdler.append(math.sqrt(sum((x - ort) ** 2 for x in s) / n))
        p_h = softmax(s)
        p_o = softmax([x / math.sqrt(d) for x in s])
        mx_h.append(max(p_h)); mx_o.append(max(p_o))
        en_h.append(entropi(p_h)); en_o.append(entropi(p_o))
    o = lambda L: sum(L) / len(L)
    print(f"{d:5d} | {o(stdler):19.2f} | {o(mx_h):14.3f} / {o(mx_o):5.3f} | {o(en_h):16.3f} / {o(en_o):5.3f}")

# --- 4) Sıra bilgisi yok: tokenleri karıştırınca çıktılar da aynı şekilde karışır ---
random.seed(1)
X = [rastgele_vektor(4) for _ in range(5)]
C1, _ = dikkat(X, X, X)
sira = [3, 0, 4, 1, 2]
C2, _ = dikkat([X[i] for i in sira], [X[i] for i in sira], [X[i] for i in sira])
fark = max(abs(a - b) for i, j in enumerate(sira) for a, b in zip(C2[i], C1[j]))
print("\nKarıştırılmış girdinin çıktısı = karıştırılmış çıktı? en büyük fark:", f"{fark:.1e}")

# --- 5) Maske sızıntı testi: gelecekteki token değişince geçmiş çıktılar değişmemeli ---
X_yeni = [v[:] for v in X]
X_yeni[4] = [10.0, -10.0, 10.0, -10.0]       # son tokeni tamamen değiştir
A, _ = dikkat(X, X, X, maske=True)
B, _ = dikkat(X_yeni, X_yeni, X_yeni, maske=True)
for i in range(5):
    print(f"  token {i}: değişim = {max(abs(a - b) for a, b in zip(A[i], B[i])):.3f}")
```

Çıktı:

```
Dikkat ağırlıkları (satır = sorgu, sütun = anahtar):
   banka  0.576  0.140  0.284
   nehir  0.140  0.576  0.284
  kenarı  0.333  0.333  0.333
Çıktılar (Value'ların ağırlıklı ortalaması):
   banka  0.718  0.282
   nehir  0.282  0.718
  kenarı  0.500  0.500
Satır toplamları: [1.0, 1.0, 1.0]
Nedensel maskeli ağırlıklar:
   banka  1.000  0.000  0.000
   nehir  0.196  0.804  0.000
  kenarı  0.333  0.333  0.333

   d | skor std (ölçeksiz) | max ağırlık ölçeksiz / ölçekli | entropi ölçeksiz / ölçekli
    4 |                1.71 |          0.490 / 0.302 |            1.446 / 1.956
   16 |                3.65 |          0.727 / 0.315 |            0.756 / 1.929
   64 |                7.45 |          0.880 / 0.327 |            0.315 / 1.913
  256 |               14.89 |          0.941 / 0.317 |            0.146 / 1.934
 1024 |               29.41 |          0.960 / 0.321 |            0.092 / 1.921

Karıştırılmış girdinin çıktısı = karıştırılmış çıktı? en büyük fark: 3.3e-16
  token 0: değişim = 0.000
  token 1: değişim = 0.000
  token 2: değişim = 0.000
  token 3: değişim = 0.000
  token 4: değişim = 11.912
```

### 4.1 Neden $\sqrt{d}$'ye Bölüyoruz?

$q$ ve $k$'nin bileşenleri ortalaması 0, varyansı 1 olan bağımsız sayılar olsun. Nokta çarpımı $q\cdot k = \sum_{c=1}^{d} q_c k_c$ tam $d$ terimin toplamıdır. Her terimin varyansı 1 olduğundan:

$$
\text{Var}(q \cdot k) = d \quad\Rightarrow\quad \text{std}(q\cdot k) = \sqrt{d}
$$

Tablodaki ikinci sütun bunu gösteriyor: $d$ 4 kat büyüdükçe skorların yayılımı yaklaşık 2 kat büyüyor ($1.7 \to 3.7 \to 7.5 \to 14.9 \to 29.4$). Softmax'e giren sayılar büyüdükçe softmax "sertleşir":

- **Ölçeksiz**, $d = 1024$'te en büyük ağırlık ortalama $0.96$ oluyor. Neredeyse her sorgu, rastgele bir anahtara **tüm dikkatini** veriyor. Entropi ($0.09$), düzgün dağılımın entropisinin ($\ln 10 = 2.30$) çok altında.
- **Ölçekli** halde $d$ ne olursa olsun en büyük ağırlık $\approx 0.32$, entropi $\approx 1.9$. Davranış boyuttan bağımsız.

Bu neden önemli? Softmax neredeyse one-hot olduğunda türevi neredeyse sıfırdır ($w_j(1 - w_j) \approx 0$). Geri yayılımla taşınan hata bu noktada söner ve model, eğitimin başında hangi tokene bakması gerektiğini **öğrenemez**. (Bu, [geri yayılım notundaki](2026-11-27-geri-yayilimi-sifirdan-yazmak-xor-ve-gradyan-kontrolu.md) sigmoid doygunluğu sorununun aynısıdır.)

**Ders 3 bağlantısı:** $\sqrt{d}$'ye bölmek, softmax'i **sıcaklığı $T = \sqrt{d}$** olacak şekilde çalıştırmakla birebir aynıdır: $\text{softmax}(s/T)$. Derste sıcaklığı çıktı kelimesini seçerken gördük. Burada aynı düğme, dikkat deseninin ne kadar "keskin" olacağını ayarlıyor ve değeri boyuta göre sabitleniyor.

### 4.2 Dikkat Sıra Bilmez

4. bölümdeki test, tokenlerin sırasını karıştırınca çıktıların da **aynı şekilde karıştığını** gösteriyor (fark $10^{-16}$, yani yalnızca yuvarlama hatası). Matematik dilinde dikkat **permütasyona eş değişkendir** (*permutation equivariant*): "köpek adamı ısırdı" ile "adam köpeği ısırdı" aynı kelime vektörlerinden oluşuyorsa, maskesiz dikkat ikisini ayırt edemez.

Bu yüzden Transformer'lar kelime gömülmelerine bir de **konum bilgisi** (*positional encoding*) ekler. Sıra, dikkat mekanizmasından değil, bu ekten gelir.

### 4.3 Maske Gerçekten Geleceği Gizliyor mu?

5. bölümde son tokeni tamamen değiştirdik ($[10, -10, 10, -10]$). Maskeli dikkatte 0–3. tokenlerin çıktıları **hiç değişmedi**, yalnızca son token etkilendi. Bu, maskenin doğru çalıştığının en basit testidir: "$j > i$ olan bir girdiyi değiştirince $c_i$ değişiyorsa, model eğitimde geleceği görüyor demektir." Bu, [veri sızıntısı notundaki](2026-11-18-veri-sizintisi-data-leakage.md) hatanın dil modellerindeki karşılığıdır: model test sırasında elinde olmayacak bir bilgiyle eğitilir.

## 5. Sık Yapılan Hatalar

**1. Softmax'i en büyük değeri çıkarmadan yazmak.** $e^{1000}$ Python'da `OverflowError` verir, NumPy'da `inf` ve ardından `nan` üretir. En büyük skoru çıkarmak sonucu değiştirmez (pay ve payda aynı $e^{-m}$ ile çarpılır) ama taşmayı önler:

```python
softmax_saf([1000.0, 999.0])   # OverflowError: math range error
softmax([1000.0, 999.0])       # [0.731, 0.269]
```

**2. Maskeyi $-\infty$ yerine $0$ ile yapmak.** $0$ "yasak" demek değildir, sıradan bir skordur: $e^0 = 1$. "nehir" satırında gelecekteki "kenarı" tokeninin skoru $0$ yapılırsa ağırlıklar $[0.164, 0.673, 0.164]$ olur, yani gelecekteki token hâlâ %16 dikkat alır. Doğrusu $-\infty$ (ya da çok büyük negatif bir sayı, örneğin $-10^9$) kullanmaktır: $e^{-\infty} = 0$.

**3. Maskeyi softmax'ten sonra uygulamak.** Softmax'ten sonra yasak ağırlıkları sıfırlarsanız satır toplamı artık 1 değildir: "nehir" için $[0.140, 0.576, 0]$, toplam $0.716$. Çıktı vektörleri küçülür ve maskesiz softmax'in paydasında gelecekteki skor hâlâ durduğundan bilgi yine sızar. Maske **softmax'ten önce, skorlara** uygulanır.

**4. Softmax'i yanlış eksende almak.** Matris kodunda `axis=0` (sütunlar) yazmak, "her anahtar dikkatini sorgulara paylaştırıyor" demektir, yanlıştır. Satır toplamlarının 1 olduğunu kontrol etmek bu hatayı hemen yakalar.

**5. $\sqrt{d}$'yi unutmak ya da $d$ ile bölmek.** Unutmak 4.1'deki doygunluğa yol açar. $d$ ile bölmek ise ters yönde aşırıya kaçar: skorlar $\sqrt{d}$ kat fazla küçülür ve desen düzgün dağılıma yaklaşır, model kimseye özellikle dikkat edemez.

**6. Büyük $q$'nun her zaman keskin dikkat verdiğini sanmak.** Keskinlik skorların **farkından** gelir. Bütün anahtarlarla aynı açıyı yapan bir sorgu (örnekteki "kenarı") ne kadar büyütülürse büyütülsün düzgün dağılım üretir (bkz. alıştırma b).

## 6. Kütüphane Karşılığı

Aynı hesabı NumPy ile matris işlemleriyle yazmak, tek döngü kullanmadan bütün sorguları birden işler:

```python
import numpy as np

def dikkat_np(Q, K, V, maske=False):
    d = K.shape[-1]
    S = Q @ K.T / np.sqrt(d)                          # n × n skor matrisi
    if maske:
        S = np.where(np.tril(np.ones(S.shape, dtype=bool)), S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)             # satır bazında taşma koruması
    W = np.exp(S)
    W /= W.sum(axis=-1, keepdims=True)                # satır toplamları = 1
    return W @ V, W
```

3. bölümdeki $Q$, $K$, $V$ ile `maske=True` çağrıldığında saf Python sürümüyle aynı ağırlıkları ($[1, 0, 0]$, $[0.196, 0.804, 0]$, $[0.333, 0.333, 0.333]$) verir. PyTorch'ta aynı işlem hazır olarak `torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)` şeklinde bulunur. Ders 3'teki **çok başlı dikkat** (*multi-head attention*), bu fonksiyonu farklı $W_Q, W_K, W_V$ matrisleriyle paralel olarak birkaç kez çalıştırıp çıktıları yan yana eklemekten ibarettir.

## 7. Alıştırma

3. bölümdeki $K$ ve $V$ matrislerini kullanın.

**a)** $\text{softmax}(s/\sqrt{d})$'nin, sıcaklığı $T = \sqrt{d}$ olan softmax ile aynı olduğunu gösterin. $d = 64$ olan bir modelde ölçekleme unutulursa, bu hangi sıcaklıkla çalışmaya denk gelir?

**b)** "kenarı" sorgusunu $[1, 1]$ yerine $[10, 10]$, "nehir" sorgusunu $[0, 1]$ yerine $[0, 10]$ yapın. Ağırlıklar nasıl değişir? Neden biri değişip diğeri değişmez?

**c)** Maskeyi $-\infty$ yerine $0$ ile uygulayan hatalı bir `dikkat` sürümü yazın ve 4.3'teki sızıntı testini ona uygulayın. Test hatayı yakalıyor mu?

<details>
<summary>Çözüm</summary>

**a)** Sıcaklıklı softmax $\text{softmax}(s/T)_j = e^{s_j/T} / \sum_{j'} e^{s_{j'}/T}$ şeklindedir. $T = \sqrt{d}$ koyunca tanım birebir aynı olur. Ölçeklemeyi unutmak, doğru modele göre sıcaklığı $\sqrt{64} = 8$ kat düşürmek ($T = 1/8$ etkisi) demektir. Ders 3'teki dille: dikkat "garantici" moda geçer ve her sorgu tek bir anahtara kilitlenir.

**b)** Çıktı:

```
kenarı [10, 10] → ağırlıklar [0.3333, 0.3333, 0.3333], çıktı [0.5, 0.5]
nehir  [0, 10]  → ağırlıklar [0.0000, 0.9992, 0.0008], çıktı [0.0004, 0.9996]
```

"kenarı" sorgusu bütün anahtarlarla aynı skoru üretir ($q\cdot k = 20$, üçü için de). Hepsini 10 ile çarpmak farkları değiştirmez (fark 0'dı, 0 kalır), dağılım düzgün kalır. "nehir" sorgusunda ise skorlar farklıdır ($[0, 1.41, 0.71]$). 10 ile çarpınca farklar da 10 kat büyür ($[0, 14.1, 7.1]$) ve softmax neredeyse one-hot olur. Softmax yalnızca skorların **farklarına** duyarlıdır. Bu yüzden $q$'nun büyüklüğü, tıpkı ters sıcaklık gibi, var olan tercihi keskinleştirir ama olmayan bir tercihi yaratamaz.

**c)**

```python
def dikkat_hatali(Q, K, V):
    d = len(K[0])
    cikti = []
    for i, q in enumerate(Q):
        skor = [nokta(q, k) / math.sqrt(d) if j <= i else 0.0   # HATA: 0, -inf değil
                for j, k in enumerate(K)]
        w = softmax(skor)
        cikti.append([sum(w[j] * V[j][c] for j in range(len(V))) for c in range(len(V[0]))])
    return cikti
```

Test hatayı yakalar. Son tokeni değiştirince 0–3. tokenlerin çıktıları da değişir:

```
  token 0: değişim = 0.968
  token 1: değişim = 0.995
  token 2: değişim = 1.787
  token 3: değişim = 1.693
  token 4: değişim = 11.912
```

 Yasak konumların skoru sabit $0$ olduğundan ağırlıkları sıfır olmaz ve son tokenin **Value**'su geçmişteki her çıktıya karışır. Değişimin sıfırdan büyük çıkması, maskenin sızdırdığını gösterir.

</details>
