# Q-Learning Adım Adım: Izgara Dünyasında Tablo Tabanlı Pekiştirmeli Öğrenme

> **Bağlantılı olduğu dersler:** Ders 2 – Grid World Etkileşimli Örnek · Ders 9 – Pekiştirmeli Öğrenme (Bellman denklemleri, keşfetme–sömürme)
> **Seviye:** Başlangıç / Orta · **Gereksinim:** Sadece standart Python (ek kütüphane yok, Python 3.8+)

## 1. Ne Problemi Çözüyoruz?

Ders 9'da ajanın amacının **toplam iskontolu getiriyi** en büyütmek olduğunu ve optimal değerlerin **Bellman optimallik denklemini** sağladığını gördük:

$$Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

Sorun şu: Bu denklemi doğrudan çözmek (değer iterasyonu gibi) için ortamın **modelini** — yani hangi eylemin hangi duruma ve ödüle götürdüğünü — bilmek gerekir. Gerçek hayatta çoğu zaman bu modeli bilmeyiz.

**Q-Learning**, modeli bilmeden, yalnızca *deneyerek* (durum → eylem → ödül → yeni durum) Q* değerlerini öğrenen **modelsiz (model-free)** ve **politika dışı (off-policy)** bir algoritmadır (Watkins, 1989).

## 2. Tek Satırlık Kalp: TD Güncellemesi

Her adımda gözlenen geçiş $(s, a, r, s')$ ile tablo şöyle güncellenir:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD hedefi}} - Q(s,a)\Big]$$

| Sembol | Anlamı | Tipik değer |
|---|---|---|
| $\alpha$ (öğrenme oranı) | Yeni bilgiye ne kadar güvenileceği | 0.05 – 0.5 |
| $\gamma$ (iskonto oranı) | Gelecekteki ödülün bugünkü değeri | 0.9 – 0.99 |
| $\varepsilon$ (keşif oranı) | Rastgele eylem seçme olasılığı | 0.05 – 0.2 |
| Köşeli parantez içi | **TD hatası**: "beklediğim" ile "gördüğüm + tahminim" arasındaki fark | — |

Sezgi: Ajan, "bu eylemden sonra en iyi davranırsam ne kazanırım?" tahminini her denemede biraz daha gerçeğe yaklaştırır. Bölüm sonu (terminal) durumlarda gelecek yoktur; hedef sadece $r$'dir.

## 3. Algoritma

1. Tüm $(s, a)$ çiftleri için $Q(s,a) = 0$ ile başla.
2. Her bölümde (episode) başlangıç durumundan başla.
3. Eylemi **ε-greedy** seç: $\varepsilon$ olasılıkla rastgele (keşfet), aksi halde $\arg\max_a Q(s,a)$ (sömür).
4. Eylemi uygula, $r$ ve $s'$'yi gözle.
5. Yukarıdaki TD güncellemesini yap.
6. $s \leftarrow s'$; terminal duruma gelene kadar 3–5'i tekrarla.
7. Yeterli bölüm sonunda politika: $\pi(s) = \arg\max_a Q(s,a)$.

> **Neden "off-policy"?** Ajan ε-greedy ile (bazen rastgele) davranır ama güncellemede $\max_{a'}$ kullanarak *açgözlü (optimal) politikanın* değerini öğrenir. Davranış politikası ile öğrenilen politika farklıdır.

## 4. Çalışan Örnek: 3×4 Izgara Dünyası

```
. . . H      H : hedef  (+1, bölüm biter)
. # . T      T : tuzak  (−1, bölüm biter)
S . . .      # : duvar,  S : başlangıç
```

Her adımın küçük bir maliyeti var (−0.04): Ajan oyalanmak yerine kısa yolu bulmaya teşvik edilir. Duvara veya ızgara dışına çarpan ajan yerinde kalır.

Kodun sonunda, öğrenilen politikayı **modeli bilen değer iterasyonunun** bulduğu optimal politika ile karşılaştırıyoruz — böylece Q-Learning'in gerçekten doğru şeyi öğrendiğini doğrulamış oluyoruz.

```python
import random

random.seed(0)

# 3x4 ızgara dünyası. S: başlangıç, H: hedef (+1), T: tuzak (-1), #: duvar
HARITA = [
    "...H",
    ".#.T",
    "S...",
]
SATIR, SUTUN = len(HARITA), len(HARITA[0])
EYLEMLER = {"↑": (-1, 0), "↓": (1, 0), "←": (0, -1), "→": (0, 1)}
ADIM_CEZASI = -0.04
GAMMA = 0.9  # iskonto oranı


def hucre(d):
    return HARITA[d[0]][d[1]]


def adim(durum, eylem):
    """Ortam: (durum, eylem) -> (yeni_durum, ödül, bitti_mi). Deterministik."""
    dr, dc = EYLEMLER[eylem]
    r, c = durum[0] + dr, durum[1] + dc
    if not (0 <= r < SATIR and 0 <= c < SUTUN) or HARITA[r][c] == "#":
        r, c = durum  # duvara/dışarı çarparsa yerinde kalır
    yeni = (r, c)
    if hucre(yeni) == "H":
        return yeni, 1.0, True
    if hucre(yeni) == "T":
        return yeni, -1.0, True
    return yeni, ADIM_CEZASI, False


DURUMLAR = [(r, c) for r in range(SATIR) for c in range(SUTUN) if HARITA[r][c] not in "#HT"]
BASLANGIC = next(d for d in DURUMLAR if hucre(d) == "S")

# ---------------- Q-Learning ----------------
Q = {(d, a): 0.0 for d in DURUMLAR for a in EYLEMLER}
ALFA, EPSILON, BOLUM = 0.1, 0.2, 3000


def en_iyi_eylem(d):
    return max(EYLEMLER, key=lambda a: Q[(d, a)])


for bolum in range(BOLUM):
    durum = BASLANGIC
    for _ in range(100):  # bölüm başına en fazla 100 adım
        # ε-greedy: ε olasılıkla keşfet, aksi halde sömür
        if random.random() < EPSILON:
            eylem = random.choice(list(EYLEMLER))
        else:
            eylem = en_iyi_eylem(durum)
        yeni, odul, bitti = adim(durum, eylem)
        hedef = odul if bitti else odul + GAMMA * max(Q[(yeni, a)] for a in EYLEMLER)
        Q[(durum, eylem)] += ALFA * (hedef - Q[(durum, eylem)])  # TD güncellemesi
        if bitti:
            break
        durum = yeni

# ---------------- Doğrulama: Değer İterasyonu (modeli bilen çözüm) ----------------
V = {d: 0.0 for d in DURUMLAR}
for _ in range(200):
    for d in DURUMLAR:
        degerler = []
        for a in EYLEMLER:
            y, o, b = adim(d, a)
            degerler.append(o if b else o + GAMMA * V[y])
        V[d] = max(degerler)


def optimal_eylem(d):
    def deger(a):
        y, o, b = adim(d, a)
        return o if b else o + GAMMA * V[y]
    return max(EYLEMLER, key=deger)


def ciz(politika):
    for r in range(SATIR):
        satir = []
        for c in range(SUTUN):
            h = HARITA[r][c]
            satir.append(h if h in "#HT" else politika((r, c)))
        print("  " + " ".join(satir))


print("Q-Learning ile öğrenilen politika:")
ciz(en_iyi_eylem)
print("\nDeğer iterasyonu (optimal) politika:")
ciz(optimal_eylem)

ayni = sum(en_iyi_eylem(d) == optimal_eylem(d) for d in DURUMLAR)
print(f"\nAynı eylemi seçen durum sayısı: {ayni}/{len(DURUMLAR)}")
print(f"Başlangıç durumunun değeri  -> Q-Learning: {max(Q[(BASLANGIC, a)] for a in EYLEMLER):.3f}"
      f" | Optimal V*: {V[BASLANGIC]:.3f}")
```

**Çıktı:**

```
Q-Learning ile öğrenilen politika:
  → → → H
  ↑ # ↑ T
  ↑ → ↑ ←

Değer iterasyonu (optimal) politika:
  → → → H
  ↑ # ↑ T
  ↑ → ↑ ←

Aynı eylemi seçen durum sayısı: 9/9
Başlangıç durumunun değeri  -> Q-Learning: 0.519 | Optimal V*: 0.519
```

### Çıktıyı Okumak

- Ajan **hiç model görmeden**, sadece 3000 bölüm deneme-yanılma ile optimal politikayı birebir buldu; başlangıç değeri de optimal $V^*$ ile aynı.
- Sağ alt köşede (`←`) ajan, yukarı çıkarsa doğrudan tuzağa düşeceğini öğrenmiş ve geri dönüyor.
- $V^*(S) = 0.519$ elle de doğrulanabilir: En kısa yol 5 adım; 4 adım cezası ve sonunda +1:
  $-0.04\,(1 + 0.9 + 0.9^2 + 0.9^3) + 0.9^4 = -0.138 + 0.656 \approx 0.519$
- Başlangıçta yukarı (`↑`) ve sağa (`→`) giden iki yol da 5 adımdır, yani **eşit değerlidir**. Kod eşitlikte ilk eylemi seçtiği için `↑` görünüyor; `→` da aynı derecede optimaldir.

## 5. Sık Yapılan Hatalar

- ❌ **Terminal durumda $\gamma \max Q(s')$ eklemek.** Bölüm bittiyse gelecek yoktur; hedef sadece $r$ olmalı. Terminal durumların Q değerleri tabloda tutuluyor ve sıfırdan farklı başlatılıyorsa (ör. rastgele veya iyimser başlatma), bu anlamsız değerler hedefe karışır ve tüm tabloya yayılır.
- ❌ **Hiç keşif yapmamak ($\varepsilon = 0$).** Q tablosu sıfırla başladığı için ajan ilk bulduğu (çoğu zaman kötü) yola takılır. Keşif–sömürü dengesi (Ders 9, Bölüm 4) şarttır; pratikte $\varepsilon$ zamanla azaltılır (ε-decay).
- ❌ **Öğrenme oranını çok büyük seçmek.** $\alpha = 1$ deterministik ortamda çalışabilir, ama ödüller/geçişler rastgele olduğunda değerler sürekli salınır. Stokastik ortamlarda küçük ve azalan $\alpha$ tercih edilir.
- ❌ **$\gamma$'yı 1 yapıp adım cezasını sıfırlamak.** Hedefe 5 adımda varmak ile 50 adımda varmak aynı değere sahip olur; ajanın kısa yolu tercih etmesi için bir sebep kalmaz.
- ❌ **Değerlendirmeyi ε-greedy ile yapmak.** Eğitim bittiğinde politikanın başarısı açgözlü ($\varepsilon = 0$) davranışla ölçülmelidir; aksi halde rastgele adımlar skoru düşürür.
- ❌ **Büyük durum uzaylarında tablo kullanmak.** Satranç veya görüntü girişli oyunlarda durum sayısı astronomiktir; tablo yerine Q fonksiyonunu bir sinir ağı ile yaklaşık hesaplamak gerekir → **DQN** (Ders 9, Bölüm 8.2).

## 6. Q-Learning ve SARSA Farkı (Kısa Not)

İkisi de tablo tabanlı TD yöntemidir; tek fark hedefte kullanılan sonraki eylemdir:

| | Hedef | Öğrendiği |
|---|---|---|
| **Q-Learning** (off-policy) | $r + \gamma \max_{a'} Q(s',a')$ | Optimal politikanın değeri |
| **SARSA** (on-policy) | $r + \gamma\, Q(s', a')$, $a'$ gerçekten seçilen eylem | Uygulanan ε-greedy politikanın değeri |

Bu yüzden SARSA, keşif sırasında tehlikeli hücrelere (uçurum, tuzak) yakın yürümekten kaçınan daha "temkinli" bir yol öğrenme eğilimindedir; Q-Learning ise en kısa ama riskli yolu öğrenir.

## 7. Mini Alıştırma

**Soru:** Ajan $s$ durumunda `→` eylemini seçiyor, $r = -0.04$ ödül alıp $s'$ durumuna geçiyor. Mevcut değerler: $Q(s, →) = 0.2$, $\max_{a'} Q(s', a') = 0.6$. $\alpha = 0.5$, $\gamma = 0.9$ ise güncelleme sonrası $Q(s, →)$ kaç olur? $s'$ terminal bir durum olsaydı (ve aynı ödül alınsaydı) sonuç ne olurdu?

<details>
<summary>Çözüm</summary>

- TD hedefi = $-0.04 + 0.9 \times 0.6 = -0.04 + 0.54 = 0.50$
- TD hatası = $0.50 - 0.2 = 0.30$
- Yeni değer = $0.2 + 0.5 \times 0.30 = $ **0.35**

$s'$ terminal olsaydı hedef sadece $r = -0.04$ olurdu:
TD hatası = $-0.04 - 0.2 = -0.24$ → yeni değer = $0.2 + 0.5 \times (-0.24) = $ **0.08**
</details>

---
*Özet:* Q-Learning, Bellman optimallik denklemini modeli bilmeden, deneyimden gelen TD hedefleriyle adım adım çözer. Keşfi (ε) unutma, terminal durumda geleceği ekleme ve sonucu açgözlü politikayla değerlendir.
