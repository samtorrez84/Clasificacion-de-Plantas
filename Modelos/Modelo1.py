import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# === Configuración ===
base_dir = r'C:\Users\samue\Documents\IPN\6IV1\Redes neuronales\PF 2\DB_dividido'
print("Existe:", os.path.exists(base_dir))

# === Generadores de datos ===
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(os.path.join(base_dir, 'train'), target_size=(128, 128), batch_size=32, class_mode='categorical')
val_generator = datagen.flow_from_directory(os.path.join(base_dir, 'val'), target_size=(128, 128), batch_size=32, class_mode='categorical')
test_generator = datagen.flow_from_directory(os.path.join(base_dir, 'test'), target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=False)

# === Modelo CNN  ===
def crear_modelo():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(15, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === Entrenamiento 5 veces ===
resultados = []
modelos = []

for i in range(5):
    print(f"\nEntrenamiento {i+1}/5")
    model = crear_modelo()
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )

    eval_result = model.evaluate(test_generator, verbose=0)
    predictions = model.predict(test_generator)
    predicted_indices = np.argmax(predictions, axis=1)

    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    resultados.append({
        'Iteración': i+1,
        'Accuracy Entrenamiento': acc,
        'Accuracy Validación': val_acc,
        'Loss Entrenamiento': loss,
        'Loss Validación': val_loss,
        'Accuracy Test': eval_result[1]
    })

    modelos.append((model, predicted_indices, history))  # guardamos también history
    model.save(f"modelo_{i+1}.h5")

# === Guardar resultados en CSV ===
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_entrenamiento.csv", index=False)
print("Resultados guardados en 'resultados_entrenamiento.csv'")

# === Mejor modelo ===
idx_mejor = df_resultados['Accuracy Test'].idxmax()
mejor_modelo, mejores_preds, mejor_history = modelos[idx_mejor]
print(f"\nMejor modelo: Iteración {idx_mejor + 1} con accuracy en test de {df_resultados.loc[idx_mejor, 'Accuracy Test']:.4f}")

# === Matriz de confusión ===
true_labels = test_generator.classes
class_names = {v: k for k, v in test_generator.class_indices.items()}

cm = confusion_matrix(true_labels, mejores_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names.values(),
            yticklabels=class_names.values(),
            cmap='Blues')
plt.xlabel('Predicha')
plt.ylabel('Real')
plt.title(f'Matriz de Confusión - Iteración {idx_mejor + 1}')
plt.tight_layout()
plt.show()

# === Reporte de Clasificación ===
print("\nReporte de clasificación del mejor modelo:")
print(classification_report(true_labels, mejores_preds, target_names=class_names.values(), zero_division=0))

# === Curvas de entrenamiento del mejor modelo ===
print(f"\nCurvas del mejor modelo (Iteración {idx_mejor + 1})")

# Precisión
plt.figure(figsize=(8, 5))
plt.plot(mejor_history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(mejor_history.history['val_accuracy'], label='Precisión validación')
plt.title('Curva de precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

# Pérdida
plt.figure(figsize=(8, 5))
plt.plot(mejor_history.history['loss'], label='Pérdida entrenamiento')
plt.plot(mejor_history.history['val_loss'], label='Pérdida validación')
plt.title('Curva de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
