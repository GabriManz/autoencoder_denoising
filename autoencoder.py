import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape):
    """
    Crea un modelo de autoencoder para la eliminación de ruido (denoising).
    
    Este modelo toma una imagen ruidosa como entrada y la reconstruye 
    eliminando el ruido utilizando una arquitectura de autoencoder con 
    capas convolucionales. El autoencoder consta de un codificador (encoder) 
    que comprime la imagen y un decodificador (decoder) que la reconstruye.

    Args:
        input_shape (tuple): La forma de la imagen de entrada (alto, ancho, canales). 
                              Por ejemplo, (256, 256, 3) para una imagen RGB de 256x256 píxeles.

    Returns:
        autoencoder (tensorflow.keras.Model): El modelo de autoencoder completo para 
                                               la eliminación de ruido.
        encoder (tensorflow.keras.Model): El modelo del codificador (encoder), útil 
                                          para obtener la representación comprimida de la imagen.
    """
    
    # Codificador (Encoder)
    input_img = layers.Input(shape=input_shape)  # Definir la entrada del modelo
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # Primera capa convolucional
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Capa de max pooling (reduce resolución)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Segunda capa convolucional
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Capa de max pooling
    encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # Capa final del codificador (más filtros)

    # Decodificador (Decoder)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)  # Capa de convolución en el decodificador
    x = layers.UpSampling2D((2, 2))(x)  # Aumentar la resolución de la imagen
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Más capas convolucionales
    x = layers.UpSampling2D((2, 2))(x)  # Aumentar nuevamente la resolución
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Más convoluciones
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Capa de salida con un solo canal (para imágenes en escala de grises)

    # Modelo de autoencoder completo
    autoencoder = models.Model(input_img, decoded)

    # Modelo de solo el codificador (útil para obtener la representación comprimida)
    encoder = models.Model(input_img, encoded)

    # Compilación del modelo
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder, encoder
