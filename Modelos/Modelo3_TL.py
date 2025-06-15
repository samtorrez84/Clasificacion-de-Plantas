import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# === Configuración del directorio ===
base_dir = r'C:\Users\samue\Documents\IPN\6IV1\Redes neuronales\PF 2\DB_dividido'
print("Existe:", os.path.exists(base_dir))

# === Generadores de datos ===
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'train'), target_size=(128, 128), batch_size=32, class_mode='categorical'
)
val_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'val'), target_size=(128, 128), batch_size=32, class_mode='categorical'
)
test_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'test'), target_size=(128, 128), batch_size=32,
    class_mode='categorical', shuffle=False
)

# Número de clases
NUM_CLASES = train_generator.num_classes

# === Crear modelo con MobileNetV2 ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Congelar capas base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predicciones = Dense(NUM_CLASES, activation='softmax')(x)

modelo = Model(inputs=base_model.input, outputs=predicciones)
modelo.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# === Entrenamiento ===
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("mejor_modelo_transfer.h5", save_best_only=True)

history = modelo.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# === Evaluación ===
eval_result = modelo.evaluate(test_generator, verbose=0)
print(f"\nAccuracy en test: {eval_result[1]:.4f}")

# === Predicciones y métricas ===
predicciones = modelo.predict(test_generator)
pred_indices = np.argmax(predicciones, axis=1)
true_labels = test_generator.classes
class_names = {v: k for k, v in test_generator.class_indices.items()}

# === Curvas de entrenamiento ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Curva de precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Curva de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()

# === Matriz de confusión ===
cm = confusion_matrix(true_labels, pred_indices)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names.values(),
            yticklabels=class_names.values(),
            cmap='Blues')
plt.xlabel('Predicha')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Transfer Learning')
plt.tight_layout()
plt.show()

# === Reporte de clasificación ===
print("\nReporte de clasificación:")
print(classification_report(true_labels, pred_indices, target_names=class_names.values(), zero_division=0))

# === Visualizar predicciones ===
filenames = test_generator.filenames
test_dir = test_generator.directory
print("Ejemplos de imágenes y predicciones:")
for i in range(5):
    path = os.path.join(test_dir, filenames[i])
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicha: {class_names[pred_indices[i]]} | Real: {class_names[true_labels[i]]}")
    plt.show()
