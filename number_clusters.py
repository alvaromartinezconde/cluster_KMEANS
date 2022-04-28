import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Cargamos el dataset (en la carpeta data con nombre Clustering)

ruta_fichero = 'inputs/Clustering.csv'
df = pd.read_csv(ruta_fichero, encoding='utf_8')

X = df[['Gol/Sh', 'npxG/Sh']].values

x_values = range(len(X))
plt.plot(x_values)

# Visualizamos la inercia

kmeans_per_k= [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1,10)]

inertias= [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8,3.5))

#plt.scatter(X[:,0], X[:,1], c=None, s=20)
plt.plot(range(1,10), inertias, "bo-")

plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)

plt.show()

# Visualizamos el sillhoute score

silhoutte_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

plt.figure(figsize=(8,3))
plt.plot(range(2,10), silhoutte_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhoutte score", fontsize=14)

plt.show()

