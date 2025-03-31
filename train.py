import yaml
from tensorflow.keras.callbacks import EarlyStopping
from autoencoder import build_autoencoder
from dataset import prepare_data

def train_autoencoder(config):
    """
    Función para entrenar el autoencoder con los datos y configuración proporcionada.
    """
    # Ruta base de los datos
    datos_dir = config['datos_dir']

    for cell in config['usar_celulas']:
        for noise_level in config['niveles_ruido']:
            noisy_dir = f"{datos_dir}/{cell}/NOISE{noise_level}"
            clean_dir = f"{datos_dir}/{cell}/NOISE0"  # Imágenes limpias en Noise0

            # Preparar los datos
            X_train, X_test, y_train, y_test = prepare_data(noisy_dir, clean_dir)

            # Verificar la forma de las imágenes antes de pasarlas al modelo
            print(f"Forma de las imágenes de entrada (X_train): {X_train.shape}")
            print(f"Forma de las imágenes de salida (y_train): {y_train.shape}")

            # Crear el modelo de autoencoder
            input_shape = X_train.shape[1:]  # Dimensiones de las imágenes de entrada
            autoencoder, encoder = build_autoencoder(input_shape)

            # Configurar early stopping para evitar el sobreajuste
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Entrenar el modelo
            autoencoder.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping]
            )

            # Guardar el modelo entrenado
            autoencoder.save(f'model/autoencoder_model_{cell}_Noise{noise_level}.h5')


def save_training_plots(history, cell, noise_level):
    """
    Guarda los gráficos de la pérdida durante el entrenamiento.
    """
    import matplotlib.pyplot as plt
    # Graficar la pérdida de entrenamiento vs la pérdida de validación
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title(f'Pérdida durante el entrenamiento - {cell} - Noise{noise_level}')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Guardar el gráfico en la carpeta 'results/'
    plt.savefig(f'results/training_loss_{cell}_Noise{noise_level}.png')
    plt.close()
