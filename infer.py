import yaml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset import load_images

def infer_on_images(config):
    """
    Función para realizar inferencias sobre nuevas imágenes utilizando el modelo entrenado.
    """
    # Ruta base de los datos
    datos_dir = config['datos_dir']

    # Obtener la ruta de las imágenes ruidosas desde el YAML
    for cell in config['usar_celulas']:
        for noise_level in config['niveles_ruido']:
            # Construir las rutas para las imágenes de cada célula y nivel de ruido
            noisy_dir = f"{datos_dir}/{cell}/NOISE{noise_level}"

            # Cargar el modelo entrenado
            model = load_model(f'model/autoencoder_model_{cell}_Noise{noise_level}.h5')

            # Cargar las imágenes ruidosas
            noisy_images = load_images(noisy_dir)

            # Normalizar las imágenes ruidosas
            noisy_images_normalized = noisy_images / 255.0

            # Realizar inferencias (predicciones)
            decoded_images = model.predict(noisy_images_normalized)

            # Mostrar las primeras 5 imágenes originales y reconstruidas
            for i in range(min(5, len(noisy_images))):
                save_inference_results(decoded_images, noisy_images, cell, noise_level, i)

def save_inference_results(decoded_images, noisy_images, cell, noise_level, index):
    """
    Guarda las imágenes originales, ruidosas y reconstruidas en la carpeta 'results/'.
    """
    plt.figure(figsize=(10, 4))

    # Imagen ruidosa
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_images[index], cmap='gray')
    plt.title("Imagen ruidosa")
    plt.axis('off')

    # Imagen reconstruida
    plt.subplot(1, 3, 2)
    plt.imshow(decoded_images[index], cmap='gray')
    plt.title("Imagen reconstruida")
    plt.axis('off')

    # Guardar el resultado en 'results/'
    plt.savefig(f'results/inference_{cell}_Noise{noise_level}_{index}.png')
    plt.close()
