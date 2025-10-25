import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Veri setii oku
df = pd.read_csv("top_100_movies_full_best_effort.csv", sep=";", skiprows=1)

# Primary Genre oluştur
df["Primary Genre"] = df["Genre(s)"].apply(lambda x: str(x).split("|")[0])

# 1. Eksik değerlerin doldurulması
imputer = SimpleImputer(strategy="median")

# Doldurulacak sayısal sütunlar
num_cols = ["IMDb Rating", "Rotten Tomatoes %", "Runtime (mins)",
            "Oscars Won", "Box Office ($M)", "Metacritic Score"]

df[num_cols] = imputer.fit_transform(df[num_cols])

#  2. Kategorik değişken kodlama
le = LabelEncoder()
df["Genre_Code"] = le.fit_transform(df["Primary Genre"])

# 3. Sayısal değişkenlerin ölçeklendirilmesi
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. Kontrol için örnek çıktı
print(df.head(5))
print("\nÖlçeklendirilmiş aralık kontrolü:")
print(df[num_cols].describe().T)
