# 🧠 Autoencoder para eliminación de ruido en imágenes de calcio

Este proyecto implementa un autoencoder convolucional en PyTorch para eliminar ruido en imágenes de fluorescencia (formato TIFF) de cardiomiocitos, preservando señales tipo sparks.

---

## 📂 Estructura del proyecto

```
autoencoder_denoise/
├── data/              ← TIFFs (no versionar)
│   └── .gitkeep
├── modelos/           ← Pesos .pth
│   └── .gitkeep
├── resultados/        ← Salidas, gráficas
│   └── .gitkeep
├── autoencoder.py     ← Definición del modelo
├── dataset.py         ← Carga de datos
├── train.py           ← Entrenamiento
├── evaluate.py        ← Evaluación y visualización
├── config.yaml        ← Parámetros de configuración
├── requirements.txt   ← Instalación vía pip
├── environment.yml    ← Instalación vía conda
├── README.md          ← Documentación del proyecto
├── LICENSE            ← Licencia del código
└── .gitignore         ← Archivos a excluir
```

---

## ⚙️ Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

---

## 🚀 Entrenamiento

Edita `config.yaml` para definir:
- Células a usar
- Niveles de ruido
- Rutas de entrada/salida
- Parámetros de entrenamiento

Luego entrena el modelo:

```bash
python train.py
```

---

## 🔍 Evaluación

Visualiza las reconstrucciones con:

```bash
python evaluate.py
```

Se comparan:
- Imagen con ruido (entrada)
- Imagen limpia (ground truth)
- Imagen reconstruida por el modelo

---

## 📓 Exploración en Jupyter

Para pruebas interactivas:

```bash
jupyter notebook demo.ipynb
```

---

## 🛠️ Características técnicas

- Compatible con imágenes no cuadradas (ej. 220×900 px)
- Entrena automáticamente en **GPU** si está disponible
- Arquitectura flexible, modular y fácil de extender

---

## 🪪 Licencia

Este proyecto está licenciado bajo la **Licencia MIT**.  
Consulta el archivo [`LICENSE`](./LICENSE) para más detalles.

---

## ✍️ Autor

Desarrollado para análisis de imágenes de fluorescencia en cardiomiocitos, con enfoque en la detección y reconstrucción de señales de calcio tipo spark.
