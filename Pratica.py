import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carregar o dataset Iris
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
data = pd.read_csv(url, header=None)

# Definir as variáveis independentes (X) e a variável dependente (y)
X = data.iloc[:, :-1].values  # As 4 características
y = data.iloc[:, -1].values   # As 3 espécies de flores

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do cotovelo para determinar o número de clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotando o gráfico do cotovelo
plt.plot(range(1, 11), inertia)
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Soma das distâncias quadráticas (Inertia)')
plt.show()

# Aplicando K-Means com k = 3 (devido ao método do cotovelo)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Visualizando os resultados (gráfico 2D)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('Clusterização K-Means')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.show()
