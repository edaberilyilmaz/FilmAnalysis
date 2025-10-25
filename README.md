# Film Verileri Üzerinde IMDb Puanı Tahminleme ve Analiz Çalışması

## 1. Veri Setinin Tanıtımı

Bu çalışmada Kaggle platformunda paylaşılan **“Top 100 Movies (Best Effort Full Data)”** veri seti kullanılmıştır.  
Veri seti, IMDb’de en yüksek puan alan 100 filmin temel özelliklerini içermektedir.  
Çalışmanın amacı, filmlerin popülerliğini ve başarılarını etkileyen faktörleri (örneğin IMDb puanı, gişe hasılatı, tür, süre, ödül sayısı vb.) incelemek ve bu değişkenler arasındaki ilişkileri analiz ederek yüksek IMDb puanına sahip filmleri öngörebilen bir model geliştirmektir.

### Veri Setindeki Değişkenler

| Değişken Adı | Açıklama |
|---------------|-----------|
| Title | Filmin adı |
| Year | Filmin vizyona girdiği yıl |
| IMDb Rating | IMDb kullanıcılarının verdiği ortalama puan (10 üzerinden) |
| Votes | IMDb’de oylama yapan kullanıcı sayısı |
| Genre(s) | Filmin tür(leri) |
| Director | Filmin yönetmeni |
| Stars / Cast | Başrol oyuncuları |
| Runtime (mins) | Filmin süresi (dakika cinsinden) |
| Oscars Won | Filmin kazandığı Oscar sayısı |
| Box Office ($M) | Gişe hasılatı (milyon dolar cinsinden) |
| Metacritic Score | Metacritic sitesindeki ortalama eleştirmen puanı |
| Rotten Tomatoes % | Rotten Tomatoes sitesindeki beğeni yüzdesi |
| Primary Genre | Filmin birincil türü (örneğin Drama, Action, Comedy) |
| Genre_Code | Türlerin sayısal karşılığı (LabelEncoder ile oluşturulmuştur) |

---

## 2. Temel Veri Analizi (Exploratory Data Analysis - EDA)

### 2.1 Eksik Değer Analizi
Veri setinde özellikle **Box Office ($M)**, **Metacritic Score** ve **Rotten Tomatoes %** sütunlarında eksik değerler tespit edilmiştir.  
Eksik değer oranı bu sütunlarda diğer değişkenlere kıyasla daha yüksektir. Bu durum modelin doğruluğunu etkileyebileceği için uygun doldurma stratejileri (örneğin ortalama, medyan veya KNN Imputer) değerlendirilmiştir.

### 2.2 Tekrarlayan Kayıt Analizi
Veri setinde mükerrer (duplicated) satırlar kontrol edilmiş, varsa kaldırılmıştır. Böylece aynı filmin birden fazla temsil edilmesi önlenmiştir.

### 2.3 Özet İstatistikler
Sayısal değişkenlerin ortalama, medyan, standart sapma ve min–max değerleri incelenmiştir.  
IMDb puanlarının ortalaması yaklaşık 8, film süreleri 120 dakika civarındadır.  
Box Office değişkeni geniş aralıkta dağılmış ve uç değerler içermektedir.

### 2.4 Kategorik Değişkenlerin Dağılımı
En sık karşılaşılan türler **Drama**, **Action** ve **Comedy**’dir.  
Bu durum tür değişkeninin IMDb puanı üzerindeki etkisini analiz etmek açısından önemlidir.

### 2.5 Sayısal Değişkenlerin Dağılımı
Histogram analizine göre:
- IMDb Rating: 8.0–8.7 aralığında yoğunlaşmıştır.  
- Rotten Tomatoes %: Filmlerin çoğu %90 üzerindedir.  
- Runtime (mins): 100–140 dakika aralığında yoğunlaşmıştır.  
- Oscars Won: Çoğu film 0’dır, yalnızca birkaç film 8–11 ödül civarındadır.  
- Box Office ($M): Sağ çarpık dağılıma sahiptir.  
- Metacritic Score: 80–100 aralığında dengeli dağılmıştır.

### 2.6 Aykırı Değer (Outlier) Analizi
Boxplot analizine göre **Box Office ($M)** değişkeninde yüksek gelirli filmler (örneğin *Titanic*, *Avatar*) aykırı değer olarak görülmektedir.  
Bu gözlemler veri hatası değildir ve modelden çıkarılmamıştır.

**Genel Değerlendirme:**  
EDA aşamasında eksik değerler, aykırı gözlemler ve değişken dengesizlikleri belirlenmiştir.  
Bu bulgular veri temizleme ve modelleme aşamalarına yön vermiştir.

---
## Görselleştirmeler
https://drive.google.com/drive/folders/1Xf5FGZdDAr1AQCE7mSwVP1Wt4-pcfqJa?usp=share_link,

---

## 3. Sayısal ve Kategorik Değişkenlerin Görsel Analizi

1. **Tür Dağılımı:** En sık görülen türler Drama, Action ve Comedy’dir.  
2. **IMDb Rating vs Box Office:** Zayıf ama pozitif ilişki gözlenmiştir.  
3. **Tür Bazında Ortalama IMDb Puanları:** Drama ve Crime türleri genellikle daha yüksek IMDb ortalamasına sahiptir.  
4. **Rotten Tomatoes % vs Metacritic Score:** Güçlü pozitif korelasyon tespit edilmiştir.  
5. **Oscars Won vs IMDb Rating:** Daha fazla Oscar kazanan filmlerin IMDb puanları da genelde yüksektir.

**Sonuç:**  
Görselleştirmeler, tür, eleştirmen puanı, ödül sayısı ve gişe başarısının IMDb puanlarıyla olan ilişkisini açıkça göstermiştir.

---

## 4. Veri Ön İşleme

### 4.1 Eksik Değerlerin Doldurulması
Eksik değerler **medyan** yöntemiyle doldurulmuştur.  
Box Office değişkeninin sağa çarpık dağılımı nedeniyle ortalamaya göre daha güvenilir bir doldurma stratejisidir.

### 4.2 Kategorik Değişkenlerin Kodlanması
Metin türündeki değişkenler **LabelEncoder** ile sayısal forma dönüştürülmüştür.

### 4.3 Sayısal Değişkenlerin Ölçeklendirilmesi
Tüm sayısal sütunlar **MinMaxScaler** yöntemiyle [0,1] aralığına ölçeklendirilmiştir.  
Bu işlem algoritmaların değişkenleri eşit önemde değerlendirmesini sağlamıştır.

---

## 5. Modelleme (Logistic Regression)

### 5.1 Modelin Amacı
Film özelliklerini kullanarak bir filmin yüksek IMDb puanına (≥8.0) sahip olup olmayacağını tahmin etmek.

### 5.2 Model Yöntemi
Model olarak **Logistic Regression** kullanılmıştır.  
Sınıf dengesizliğini azaltmak için `class_weight="balanced"` parametresi eklenmiştir.  
Veri %75 eğitim ve %25 test oranında bölünmüş, 5-katlı **cross-validation** uygulanmıştır.

### 5.3 Girdi Değişkenleri
IMDb Rating, Rotten Tomatoes %, Runtime (mins), Oscars Won, Box Office ($M), Metacritic Score, Genre_Code

### 5.4 Model Değerlendirme Sonuçları

| Metrik | Değer |
|:-------|:------:|
| Accuracy (Doğruluk) | 0.84 |
| Precision (Kesinlik) | 0.25 |
| Recall (Duyarlılık) | 0.50 |
| F1-Score | 0.33 |
| ROC-AUC | 0.91 |

### 5.5 ROC Eğrisi Yorumu
ROC eğrisi (AUC = 0.91), modelin sınıfları ayırt etme gücünün yüksek olduğunu göstermektedir.  
Pozitif sınıf (yüksek IMDb puanlı filmler) başarılı biçimde tespit edilmiştir.

### 5.6 Korelasyon Analizi
Sayısal değişkenler arasındaki ilişkiler incelenmiştir:  
- Rotten Tomatoes % ile Metacritic Score arasında güçlü pozitif korelasyon (r = 0.76)  
- IMDb Rating ile Box Office ($M) arasında orta düzey ilişki (r ≈ 0.40)  
- Oscars Won ile Box Office ($M) arasında zayıf pozitif ilişki (r = 0.35)

Yüksek düzeyde multicollinearity bulunmamaktadır; veri Logistic Regression modeli için uygundur.

### 5.7 Sonuçların Değerlendirilmesi
Model küçük ve dengesiz bir veri setinde anlamlı sonuçlar vermiştir:  
- Accuracy: %84  
- ROC-AUC: 0.91  
- F1-score: 0.33  

Model genel eğilimi yakalamakta başarılıdır.  
Daha büyük ve dengeli veri setleriyle performans artışı beklenmektedir.

---

## 6. Sonuçların Yorumlanması

Yapılan analizler sonucunda Logistic Regression modeli %84 doğruluk ve 0.91 ROC-AUC değeri elde etmiştir.  
Bu değerler, modelin sınıfları genel olarak doğru biçimde ayırt ettiğini göstermektedir.  

F1-score’un düşük kalması, veri setinde yüksek IMDb puanlı filmlerin az olmasından kaynaklanmıştır.  
Model genel eğilimi doğru yakalamış olsa da sınıf dengesizliği performansı sınırlandırmıştır.  

Bu proje kapsamında veri temizleme, görselleştirme, korelasyon analizi ve modelleme adımları uygulanmış,  
veri bilimi sürecinin tüm aşamaları pratik olarak deneyimlenmiştir.  
Ayrıca, küçük ve dengesiz veri setlerinde yalnızca doğruluk oranına değil,  
**F1-score** ve **ROC-AUC** gibi tamamlayıcı metriklere de bakılması gerektiği öğrenilmiştir.

---

