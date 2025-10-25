import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("top_100_movies_full_best_effort.csv", sep=";", skiprows=1)

# Primary Genre sütunu ekle
df["Primary Genre"] = df["Genre(s)"].apply(lambda x: str(x).split("|")[0])

# 1. Tür dağılımı
plt.figure(figsize=(8,4))
df["Genre(s)"].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title("En Sık Görülen Türler")
plt.xlabel("Tür")
plt.ylabel("Film Sayısı")
plt.tight_layout()
plt.show()

# 2. IMDb Rating vs Box Office
plt.figure(figsize=(6,4))
plt.scatter(df["IMDb Rating"], df["Box Office ($M)"], alpha=0.6)
plt.title("IMDb Rating vs Box Office ($M)")
plt.xlabel("IMDb Rating")
plt.ylabel("Box Office ($M)")
plt.tight_layout()
plt.show()

# 3. Tür bazında ortalama IMDb puan
plt.figure(figsize=(8,4))
df.groupby("Primary Genre")["IMDb Rating"].mean().sort_values(ascending=False).plot(kind='bar', color='lightgreen')
plt.title("Tür Bazında Ortalama IMDb Puanları")
plt.ylabel("Ortalama IMDb Rating")
plt.tight_layout()
plt.show()

# 4. Roten Tomatoes vs Metacritic
print(df.columns.tolist())  # sütun isimlerini kontrol et
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="Rotten Tomatoes %", y="Metacritic Score", alpha=0.7)
plt.title("Rotten Tomatoes % vs Metacritic Score")
plt.tight_layout()
plt.show()

# 5. OscarsWon vs IMDb Rating
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x="Oscars Won", y="IMDb Rating")
plt.title("Oscars Won vs IMDb Rating")
plt.tight_layout()
plt.show(block=True)


