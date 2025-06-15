import os
import shutil
from sklearn.model_selection import train_test_split

# Ruta de la carpeta"
original_dataset_dir = r'C:\Users\samue\Documents\IPN\6IV1\Redes neuronales\PF 2\DB'
print("Existe:", os.path.exists(original_dataset_dir))

base_dir = os.path.join(original_dataset_dir + '_dividido')
os.makedirs(base_dir, exist_ok=True)

# Filtra solo carpetas válidas de clases
clases = [d for d in os.listdir(original_dataset_dir)
          if os.path.isdir(os.path.join(original_dataset_dir, d)) and not d.startswith('$')]

print("Clases encontradas:", clases)

# División y copiado
for clase in clases:
    ruta_clase = os.path.join(original_dataset_dir, clase)
    img_paths = [os.path.join(ruta_clase, fname)
                 for fname in os.listdir(ruta_clase)
                 if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 80% train_val, 20% test
    train_val, test = train_test_split(img_paths, test_size=0.2, random_state=42)
    # de train_val: 80% train, 20% val (que es 16% del total)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    for split_name, split_set in zip(['train', 'val', 'test'], [train, val, test]):
        split_clase_dir = os.path.join(base_dir, split_name, clase)
        os.makedirs(split_clase_dir, exist_ok=True)
        for path in split_set:
            shutil.copy(path, split_clase_dir)

print("División completada. Carpeta base:", base_dir)
