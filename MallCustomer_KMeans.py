# Gerekli kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Veri setini oku
veriler = pd.read_csv(
    '/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'
)

# Sayısal değişkenleri seç (Age, Income, Spending Score)
X = veriler.iloc[:, 2:].values

# -----------------------------
# 1. VERİ ÖLÇEKLENDİRME
# -----------------------------
sc = StandardScaler()
X = sc.fit_transform(X)

# -----------------------------
# 2. ELBOW METHOD
# -----------------------------
sonuclar = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=10,
        random_state=123
    )
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), sonuclar)
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


# -----------------------------
# 3. FINAL KMEANS (k = 4)
# -----------------------------
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    n_init=10,
    random_state=123
)

y_pred = kmeans.fit_predict(X)

# Küme bilgisini veri setine ekle
veriler['Cluster'] = y_pred

# -----------------------------
# 4. KÜME MERKEZLERİ
# -----------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = sc.inverse_transform(centers_scaled)

print("Küme Merkezleri (Age, Income, Spending Score):")
print(centers_original)

# -----------------------------
# 5. GÖRSELLEŞTİRME
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(
    veriler['Annual Income (k$)'],
    veriler['Spending Score (1-100)'],
    c=veriler['Cluster']
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation (K=4)')
plt.show()
