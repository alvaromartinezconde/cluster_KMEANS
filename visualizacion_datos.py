import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


# Cargamos el dataset (en la carpeta data con nombre Clustering)

ruta_fichero = 'inputs/Clustering.csv'
df = pd.read_csv(ruta_fichero, encoding='utf_8')


# Escalar las variables Gol/Sh y npxG/Sh

def PreparingData(dt):

  df[['Gol/Sh','npxG/Sh']] = StandardScaler().fit_transform(dt[['Gol/Sh','npxG/Sh']])

  return dt

test_set_prep_3 = PreparingData(df)

# Visualizamos los datos

X = df[['Gol/Sh', 'npxG/Sh']].values

plt.figure(figsize=(16,8))

plt.scatter(X[:,0], X[:,1], c=None, s=20)

plt.xlabel("Gol/Sh", fontsize=14)
plt.ylabel("npxG/Sh", fontsize=14, rotation=0)
plt.show()
