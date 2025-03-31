import yaml
from train import train_autoencoder
from evaluate import evaluate_model
from infer import infer_on_images

# Cargar la configuración desde el archivo YAML
def load_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def run_pipeline():
    """
    Ejecuta el flujo completo del proyecto: entrenamiento, evaluación e inferencia.
    """
    # Cargar configuración desde el archivo YAML
    config = load_config()

    # Entrenar el modelo
    print("1. Entrenando el modelo...")
    train_autoencoder(config)  # Pasamos la configuración a la función

    # Evaluar el modelo
    print("2. Evaluando el modelo...")
    evaluate_model(config)  # Pasamos la configuración a la función

    # Realizar inferencia sobre nuevas imágenes
    print("3. Realizando inferencia sobre nuevas imágenes...")
    infer_on_images(config)  # Pasamos la configuración a la función

if __name__ == "__main__":
    run_pipeline()  # Ejecutar el pipeline completo
