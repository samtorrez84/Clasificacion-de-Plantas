import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.utils import image_dataset_from_directory

#parametros iniciales
data_dir = r"C:\Users\samas\OneDrive\Documents\5to semestre\proyecto plantas\Clasificacion-de-Plantas\DB"
img_size = (64, 64)  # Tamaño pequeño para t-SNE rápido
batch_size = 32

#carga del dataset
dataset = image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = dataset.class_names

X_all = []
y_all = []

for batch, labels in dataset:
    X_all.append(batch.numpy())
    y_all.append(labels.numpy())

X_all = np.concatenate(X_all) 
y_all = np.concatenate(y_all)

X_flat = X_all.reshape((X_all.shape[0], -1)) 

#Aplicar el tsne
print("Calculando t-SNE sobre las imágenes...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_flat)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=[class_names[i] for i in y_all],
    palette="hsv",
    s=50,
    legend="full"
)
plt.title("t-SNE aplicado directamente al dataset de imágenes (64x64)")
plt.xlabel("Componente t-SNE 1")
plt.ylabel("Componente t-SNE 2")
plt.savefig("tsne_dataset.png")
plt.show()
