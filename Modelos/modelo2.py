import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


#parametros generales
data_dir = r"C:\Users\samas\OneDrive\Documents\5to semestre\proyecto plantas\Clasificacion-de-Plantas\DB"
img_size = (128, 128)
batch_size = 32
epochs = 20
num_classes = len(next(os.walk(data_dir))[1])
print(f"Detectadas {num_classes} clases.")

#Carga del dataset
print("Cargando dataset...")
dataset = image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=42
)
dataset = dataset.map(lambda x, y: (x / 255.0, y))

#División de batches
total_batches = tf.data.experimental.cardinality(dataset).numpy()
train_batches = int(0.7 * total_batches)
val_batches = int(0.15 * total_batches)

train_ds = dataset.take(train_batches)
val_ds = dataset.skip(train_batches).take(val_batches)
test_ds = dataset.skip(train_batches + val_batches)

#Modelo CNN2 
def build_model_b():
    model = models.Sequential([
        layers.InputLayer(input_shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.25),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

results = []
mejor_precision = -1
mejor_iteracion = -1

for i in range(5):
    print(f"\n=== Entrenamiento de la iteración {i+1} ===")
    model = build_model_b()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stop]
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    results.append({'Iteración': i+1, 'Pérdida de prueba': test_loss, 'Precisión de prueba': test_acc})

    if test_acc > mejor_precision:
        mejor_precision = test_acc
        mejor_iteracion = i + 1
        model.save("modelo_mejor_iteracion.h5")
        print(f"Nuevo mejor modelo guardado: 'modelo_mejor_iteracion.h5' con precisión {test_acc:.4f}")


df_resultados = pd.DataFrame(results)
df_resultados.to_csv("resultados_modeloB.csv", index=False)
print("\nResultados guardados en 'resultados_modeloB.csv'.")

#Resultados dde las iteraciones
media_loss = df_resultados['Pérdida de prueba'].mean()
varianza_loss = df_resultados['Pérdida de prueba'].var()
std_loss = df_resultados['Pérdida de prueba'].std()

media_acc = df_resultados['Precisión de prueba'].mean()
varianza_acc = df_resultados['Precisión de prueba'].var()
std_acc = df_resultados['Precisión de prueba'].std()

resumen = pd.DataFrame({
    'Métrica': ['Pérdida', 'Precisión'],
    'Media': [media_loss, media_acc],
    'Desviación estándar': [std_loss, std_acc],
    'Varianza': [varianza_loss, varianza_acc]
})
resumen.to_csv("resumen_estadistico.csv", index=False)
print("\nResumen guardado en 'resumen_estadistico.csv'.")
print(resumen.to_string(index=False))
print(f"\n El mejor modelo fue el de la iteración {mejor_iteracion} con precisión {mejor_precision:.4f}")

# Precisión
plt.figure(figsize=(8, 5))
plt.plot(df_resultados['Iteración'], df_resultados['Precisión de prueba'], marker='o', label='Precisión')
plt.axhline(media_acc, color='green', linestyle='--', label='Media')
plt.title('Precisión por Iteración')
plt.xlabel('Iteración')
plt.ylabel('Precisión')
plt.grid(True)
plt.legend()
plt.savefig("grafica_precision.png")
plt.show()

# Pérdida
plt.figure(figsize=(8, 5))
plt.plot(df_resultados['Iteración'], df_resultados['Pérdida de prueba'], marker='o', color='red', label='Pérdida')
plt.axhline(media_loss, color='orange', linestyle='--', label='Media')
plt.title('Pérdida por Iteración')
plt.xlabel('Iteración')
plt.ylabel('Pérdida')
plt.grid(True)
plt.legend()
plt.savefig("grafica_perdida.png")
plt.show()

#Matriz de confusión
print("\nGenerando matriz de confusión del mejor modelo...")

mejor_modelo = tf.keras.models.load_model("modelo_mejor_iteracion.h5")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = mejor_modelo.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
labels = list(range(num_classes))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusión - Mejor Modelo')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.savefig("matriz_confusion2.png")
plt.show()

print("Matriz de confusión guardada como 'matriz_confusion2.png'")
