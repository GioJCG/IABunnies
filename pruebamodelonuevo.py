import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo guardado
model = tf.keras.models.load_model('reconocimiento-rostro/1/', compile=False)

# Asegúrate de que la dimensión de entrada coincida con tus imágenes reales
input_shape = (150, 150, 1)

# Configura el modelo para compilación
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ... Resto del código de carga y predicción


# Lista de personas (debes mantenerla consistente con la que usaste durante el entrenamiento)
listaPersonas = ['Daniela', 'Gerardo', 'Jazael']

# Inicializar el LabelEncoder y ajustarlo a la lista de personas
label_encoder = LabelEncoder()
label_encoder.fit(listaPersonas)

# Definir una función para preprocesar la imagen
def preprocess_image(image_path):
    # Cargar la imagen
    img = Image.open(image_path)

    # Redimensionar la imagen a 150x150
    img = img.resize((150, 150))

    # Convertir la imagen a escala de grises si es necesario
    img = img.convert('L')

    # Convertir la imagen a un array de NumPy
    img_array = np.array(img)

    # Normalizar los valores de píxeles
    img_array = img_array / 255.0

    # Agregar dimensiones para el canal y el lote
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Ruta de la imagen que deseas probar
image_path = 'path/to/your/test/image.jpg'

# Preprocesar la imagen
input_image = preprocess_image(image_path)

# Realizar la predicción
predictions = model.predict(input_image)

# Decodificar las predicciones
predicted_class_index = np.argmax(predictions)
predicted_class = label_encoder.classes_[predicted_class_index]

# Imprimir los resultados
print(f'Clase predicha: {predicted_class}')
print(f'Probabilidades: {predictions}')
