import os
import tifffile as tiff
import numpy as np
from sklearn.model_selection import train_test_split

# Función para cargar imágenes de un directorio
def load_images(image_dir):
    """
    Carga todas las imágenes TIFF de un directorio y las adapta a la forma correcta.
    """
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".tif"):
            img_path = os.path.join(image_dir, filename)
            image = tiff.imread(img_path)
            # Asegurarse de que la imagen tenga la forma (alto, ancho, 1) para que sea compatible con Conv2D
            image = np.expand_dims(image, axis=-1)  # Añadir la dimensión del canal (debe ser 1 para imágenes en escala de grises)
            images.append(image)
    return np.array(images)

def prepare_data(noisy_dir, clean_dir):
    """
    Prepara los datos para entrenamiento y evaluación, asegurándose de que las imágenes tienen la forma correcta.
    """
    # Cargar las imágenes limpias (NOISE0)
    clean_images = load_images(clean_dir)

    # Cargar las imágenes ruidosas
    noisy_images = load_images(noisy_dir)
    
    # Asegurarse de que las imágenes ruidosas y limpias tengan el mismo tamaño
    assert noisy_images.shape == clean_images.shape, "Las imágenes ruidosas y limpias deben tener el mismo tamaño."

    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(noisy_images, clean_images, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
