import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Ruta de la imagen de prueba en formato original (no en base64)
sample_image_path = 'data/Jazael/rostro_0.jpg'

# Lee la imagen
img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

# Preprocesa la imagen para que tenga el mismo formato que las imágenes de entrenamiento
resized_img = cv2.resize(img, (150, 150))
resized_img = resized_img / 255.0  # Normaliza los valores de píxeles al rango [0, 1]

# Expande las dimensiones para que coincida con el formato de entrada esperado por el modelo
input_data = np.expand_dims(resized_img, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

# Carga el modelo guardado
model_path = 'reconocimiento-rostro/1/'
loaded_model = tf.keras.models.load_model(model_path)

# Realiza la predicción
prediction = loaded_model.predict(input_data)

# Imprime la predicción
print(prediction)