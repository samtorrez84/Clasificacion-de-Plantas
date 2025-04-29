from torchvision import transforms
from PIL import Image
import os

# Rutas
carpeta_origen = 'imagenes_originales'
carpeta_destino = 'imagenes_aumentadas'

# Asegurarse que la carpeta destino exista
os.makedirs(carpeta_destino, exist_ok=True)

# Definir transformaciones individuales
transformaciones = {
    'flip_horizontal': transforms.RandomHorizontalFlip(p=1.0),
    'rotacion': transforms.RandomRotation(20),
    'color_jitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
}

# Listar todas las imágenes en la carpeta origen
imagenes = [img for img in os.listdir(carpeta_origen) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

contador = 0

for imagen_nombre in imagenes:
    # Cargar imagen original
    ruta_imagen = os.path.join(carpeta_origen, imagen_nombre)
    imagen = Image.open(ruta_imagen).convert('RGB')

    # Aplicar cada transformación por separado
    for nombre_transf, transformacion in transformaciones.items():
        imagen_aumentada = transformacion(imagen)
        
        # Guardar imagen transformada
        nombre_base, extension = os.path.splitext(imagen_nombre)
        nuevo_nombre = f"{nombre_base}_{nombre_transf}{extension}"
        ruta_guardado = os.path.join(carpeta_destino, nuevo_nombre)
        imagen_aumentada.save(ruta_guardado)
        
        contador += 1

print(f"¡Listo! Se guardaron {contador} imágenes aumentadas en la carpeta: {carpeta_destino}")
