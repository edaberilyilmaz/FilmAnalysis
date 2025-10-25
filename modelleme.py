import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay


#  Veri setini oku
df = pd.read_csv("top_100_movies_full_best_effort.csv", sep=";", skiprows=1)

#  Primary Genre sütununu oluştur
df["Primary Genre"] = df["Genre(s)"].apply(lambda x: str(x).split("|")[0])

#  Eksik değerleri doldur (median)
num_cols = ["IMDb Rating", "Rotten Tomatoes %", "Runtime (mins)",
            "Oscars Won", "Box Office ($M)", "Metacritic Score"]

imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

#  Kategorik sütun kodlama
le = LabelEncoder()
df["Genre_Code"] = le.fit_transform(df["Primary Genre"])

#  Sayısal değişkenleri ölçekle
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#  Hedef değişken (dengeli versiyon)
# IMDb Rating artık 0–1 aralığında, bu yüzden 0.7 eşiği IMDb≈8.0’a denk gelir
df["target"] = (df["IMDb Rating"] >= 0.7).astype(int)

#  Özellikler ve hedefi ayır
X = df[num_cols + ["Genre_Code"]]
y = df["target"]

#  Train-Test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#  Model oluşturma ve eğitme
model = LogisticRegression(max_iter=500, C=1.0, penalty="l2", solver="lbfgs", class_weight="balanced")
model.fit(X_train, y_train)

#  Tahminler
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

#  Performans raporu
print("=== Tek Test Sonuçları ===")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ROC eğrisi
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - Logistic Regression")
plt.show()

#  Cross-Validation (5 katlı)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])

print("\n=== 5-Katlı Cross-Validation Ortalama Sonuçları ===")
print(f"Accuracy: {scores['test_accuracy'].mean():.3f}")
print(f"F1-Score: {scores['test_f1'].mean():.3f}")
print(f"ROC-AUC:  {scores['test_roc_auc'].mean():.3f}")
