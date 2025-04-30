import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

base_dir = r'C:\Users\samue\Documents\IPN\6IV1\Redes neuronales\PF 2\DB_dividido'
print("Existe:", os.path.exists(base_dir))

# Normalización 
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'train'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'val'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'test'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)






from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # número de clases
])



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)




# === PREDICCIONES ===
predictions = model.predict(test_generator)
predicted_indices = np.argmax(predictions, axis=1)

# === ETIQUETAS REALES ===
true_labels = test_generator.classes

# === NOMBRES DE CLASES ===
class_indices = test_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}

# === MOSTRAR PREDICCIÓN y REAL ===
print("🔍 Primeras 10 predicciones:")
for i in range(10):
    pred_class = class_names[predicted_indices[i]]
    true_class = class_names[true_labels[i]]
    print(f"Imagen {i+1}: Predicha = {pred_class} | Real = {true_class}")

# === MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(true_labels, predicted_indices)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names.values(),
            yticklabels=class_names.values(),
            cmap='Blues')
plt.xlabel('Predicha')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()

# === REPORTE DE CLASIFICACIÓN ===
print("\n📋 Reporte de clasificación:")
print(classification_report(true_labels, predicted_indices, target_names=class_names.values()))

# === Mostrar imágenes con etiquetas ===
import matplotlib.image as mpimg

filenames = test_generator.filenames
test_dir = test_generator.directory

print("\n🖼️ Visualización de imágenes con etiquetas predichas:")
for i in range(5):
    path = os.path.join(test_dir, filenames[i])
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicha: {class_names[predicted_indices[i]]} | Real: {class_names[true_labels[i]]}")
    plt.show()





# === PRECISIÓN ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

# === PÉRDIDA ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()


"""model.save("10epochs.h5")
model.save("10epochs.keras")"""

