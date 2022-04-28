import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargamos el dataset (en la carpeta data con nombre Clustering)

ruta_fichero = 'inputs/Clustering.csv'
df = pd.read_csv(ruta_fichero, encoding='utf_8')


def Exec_KMeans(k, X):
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    print("Centros de los clusters: ", kmeans.cluster_centers_)
    
    plt.figure(figsize=(16, 6))
    cl = ['red', 'green', 'blue', 'yellow', 'black']
    assign = []
    for row in y_pred:
        assign.append(cl[row])
        
    plt.scatter(X[:,0], X[:,1], c=assign, s=70)
    
    plt.show()
    
    return y_pred


    # Ejecutamos KMeans con el número de clusters (K) que hayas elegido y guardalo en una variable 'Cluster'
# en el dataset original (dt, si es tu caso)

K=2

X = df[['Gol/Sh', 'npxG/Sh']].values

Cluster=Exec_KMeans(K,X)

Cluster

# DEJAMOS SELECCIONADO EL CLUSTER 0


selected_k=2

df['Cluster'] = Exec_KMeans(selected_k,X)

df.head()

dfcluster = df.loc[:, 'Cluster'] == 0

df[dfcluster].count()
