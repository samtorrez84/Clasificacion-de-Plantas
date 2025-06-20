import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns

#parametros
data_dir = r"C:\Users\samas\OneDrive\Documents\5to semestre\proyecto plantas\Clasificacion-de-Plantas\DB"
img_size = (224, 224) 
batch_size = 32
epochs = 20
num_classes = len(next(os.walk(data_dir))[1])
print(f"Detectadas {num_classes} clases.")

#carga del dataset
print("Cargando dataset...")
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

#Normalización automática para EfficientNet
preprocess_input = tf.keras.applications.efficientnet.preprocess_input
dataset = dataset.map(lambda x, y: (preprocess_input(x), y))

#división de batches
total_batches = tf.data.experimental.cardinality(dataset).numpy()
train_batches = int(0.7 * total_batches)
val_batches = int(0.15 * total_batches)

train_ds = dataset.take(train_batches)
val_ds = dataset.skip(train_batches).take(val_batches)
test_ds = dataset.skip(train_batches + val_batches)

#modelo con EFFICIENTNETB0 
def build_transfer_model():
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # Congela todas las capas inicialmente

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

results = []
best_acc = -1
best_iter = -1

for i in range(5):
    print(f"\n=== Entrenamiento de la iteración {i+1} ===")
    model = build_transfer_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stop]
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    results.append({'Iteración': i+1, 'Pérdida de prueba': test_loss, 'Precisión de prueba': test_acc})

    if test_acc > best_acc:
        best_acc = test_acc
        best_iter = i + 1
        model.save("mejor_modelo_transfer.h5")
        print(f"Mejor modelo actualizado (iteración {i+1}) con precisión {test_acc:.4f}")

#Resultados de las iteraciones
df = pd.DataFrame(results)
df.to_csv("resultados_transfer.csv", index=False)

media_loss = df['Pérdida de prueba'].mean()
media_acc = df['Precisión de prueba'].mean()
std_loss = df['Pérdida de prueba'].std()
std_acc = df['Precisión de prueba'].std()
var_loss = df['Pérdida de prueba'].var()
var_acc = df['Precisión de prueba'].var()

resumen = pd.DataFrame({
    'Métrica': ['Pérdida', 'Precisión'],
    'Media': [media_loss, media_acc],
    'Desviación estándar': [std_loss, std_acc],
    'Varianza': [var_loss, var_acc]
})
resumen.to_csv("resumen_transfer.csv", index=False)
print("\nResumen estadístico guardado.")

#Matriz de confusión
print("Generando matriz de confusión del mejor modelo...")

modelo_final = tf.keras.models.load_model("mejor_modelo_transfer.h5")

y_true, y_pred = [], []
for x_batch, y_batch in test_ds:
    preds = modelo_final.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión - Transfer Learning")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.savefig("matriz_confusion_transfer.png")
plt.show()
