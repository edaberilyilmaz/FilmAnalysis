import pandas as pd
import matplotlib.pyplot as plt


# 1. Veri setini oku
df = pd.read_csv("top_100_movies_full_best_effort.csv", sep=";", skiprows=1)

# 2. Eksik değer analizi
missing_count = df.isna().sum()
missing_ratio = (df.isna().mean() * 100).round(2)
missing_df = pd.DataFrame({
    "Eksik Değer Sayısı": missing_count,
    "Eksik Değer Oranı (%)": missing_ratio
}).sort_values("Eksik Değer Oranı (%)", ascending=False)
print(missing_df)

# 3. Tekrarlayan kayıt kontrolü
duplicates = df.duplicated().sum()
print(f"Tekrarlayan kayıt sayısı: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)

# 4. Özet istatistikler
print(df.describe(include="all").T)

# 5. Kategorik değişkenlerin dağılımı
print(df["Genre(s)"].value_counts().head(10))

# 6. Sayısal değişkenlerin dağılımı
num_cols = ["IMDb Rating", "Rotten Tomatoes %", "Runtime (mins)", "Oscars Won", "Box Office ($M)", "Metacritic Score"]
df[num_cols].hist(bins=10, figsize=(10,6))
plt.suptitle("Sayısal Değişkenlerin Dağılımı", y=1.02)
plt.tight_layout()
plt.show()

# 7. Aykırı değer (boxplot)
plt.figure(figsize=(8,5))
df.boxplot(column=["Box Office ($M)", "Runtime (mins)", "Oscars Won"])
plt.title("Aykırı Değer Analizi (Boxplot)")
plt.show()



