import yaml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset import prepare_data

def evaluate_model(config):
    """
    Función para evaluar el modelo con los datos y configuración proporcionada.
    """
    # Ruta base de los datos
    datos_dir = config['datos_dir']

    # Obtener las rutas de las imágenes ruidosas y limpias desde el YAML
    for cell in config['usar_celulas']:
        for noise_level in config['niveles_ruido']:
            # Construir las rutas para las imágenes de cada célula y nivel de ruido
            noisy_dir = f"{datos_dir}/{cell}/NOISE{noise_level}"
            clean_dir = f"{datos_dir}/{cell}/NOISE0"  # Imágenes limpias en Noise0

            # Preparar los datos
            X_train, X_test, y_train, y_test = prepare_data(noisy_dir, clean_dir)

            # Cargar el modelo entrenado
            model = load_model(f'model/autoencoder_model_{cell}_Noise{noise_level}.h5')

            # Evaluar el modelo
            loss = model.evaluate(X_test, y_test)
            print(f"Loss para {cell} - Noise{noise_level}: {loss}")

            # Guardar las métricas de evaluación
            save_evaluation_metrics(loss, cell, noise_level)

            # Visualizar y guardar las imágenes originales, ruidosas y reconstruidas
            decoded_images = model.predict(X_test)
            for i in range(5):
                save_inference_results(decoded_images, X_test, y_test, cell, noise_level, i)

def save_inference_results(decoded_images, noisy_images, clean_images, cell, noise_level, index):
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

    # Imagen original
    plt.subplot(1, 3, 3)
    plt.imshow(clean_images[index], cmap='gray')
    plt.title("Imagen original")
    plt.axis('off')

    # Guardar el resultado en 'results/'
    plt.savefig(f'results/inference_{cell}_Noise{noise_level}_{index}.png')
    plt.close()

def save_evaluation_metrics(loss, cell, noise_level):
    """
    Guarda las métricas de evaluación en un archivo CSV dentro de la carpeta 'results/'.
    """
    import csv
    with open('results/evaluation_metrics.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cell, noise_level, loss])
