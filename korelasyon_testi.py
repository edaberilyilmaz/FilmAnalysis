import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv("top_100_movies_full_best_effort.csv", sep=";", skiprows=1)
df["Primary Genre"] = df["Genre(s)"].apply(lambda x: str(x).split("|")[0])

# KORELASYON ANALİZİ
corr_matrix = df[["IMDb Rating", "Rotten Tomatoes %", "Runtime (mins)",
                  "Oscars Won", "Box Office ($M)", "Metacritic Score"]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Değişkenler Arası Korelasyon Analizi")
plt.show()

