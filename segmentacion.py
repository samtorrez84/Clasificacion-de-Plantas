"""
# === 0. Instalación de dependencias y descarga del modelo SAM ===
!pip install opencv-python matplotlib torch torchvision
!git clone https://github.com/facebookresearch/segment-anything.git
%cd segment-anything
!pip install -e .

# Descargar el checkpoint del modelo SAM
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P /content/
"""

# === 1. Importar librerías necesarias ===
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

# === 2. Configuración de carpetas ===
input_folder = "/content/77/"
output_folder = "/content/segmentacion/77/"
os.makedirs(output_folder, exist_ok=True)

# === 3. Cargar modelo SAM ===
sam_checkpoint = "/content/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# === 4. Listar todas las imágenes en la carpeta de entrada ===
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Ordenar opcionalmente
image_files.sort()

print(f"🔍 Encontradas {len(image_files)} imágenes para procesar.")

# === 5. Procesar cada imagen ===
for idx, image_name in enumerate(image_files):
    print(f"Procesando imagen {idx+1}/{len(image_files)}: {image_name}")
    
    image_path = os.path.join(input_folder, image_name)

    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar {image_name}, saltando.")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Setear imagen en SAM
    predictor.set_image(image)

    # Definir 3 puntos (centro, ±50 pixeles horizontalmente)
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    input_point = np.array([
        [center_x, center_y],
        [center_x + 50, center_y],
        [center_x - 50, center_y],
    ])

    input_label = np.array([1, 1, 1])

    # Predecir máscara
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask = masks[0]

    # Aplicar máscara
    binary_mask = (best_mask * 255).astype(np.uint8)

    if binary_mask.shape[:2] != image.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

    # Redimensionar a 128x128
    final_image = cv2.resize(masked_image, (128, 128), interpolation=cv2.INTER_AREA)

    # Crear nombre nuevo basado en original
    base_name = os.path.splitext(image_name)[0]
    new_name = f"{base_name}_segmentado.png"
    output_path = os.path.join(output_folder, new_name)

    # Guardar
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"Imagen procesada y guardada en: {output_path}")

print(f"Todas las imágenes procesadas y guardadas en: {output_folder}")
