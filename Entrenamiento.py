import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

images = []
labels = []
listaPersonas = [
    'Daniela',
    'Gerardo',
    'Jazael',
    'Melissa',
]

dataPath = f'{os.getcwd()}/data'

for nombrePersona in listaPersonas:
    rostrosPath = dataPath + '/' + nombrePersona
    for filename in os.listdir(rostrosPath):
        img_path = os.path.join(rostrosPath, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(nombrePersona)

images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(listaPersonas), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape(-1, 150, 150, 1), y_train, epochs=10, validation_data=(X_test.reshape(-1, 150, 150, 1), y_test))

model.save('model/modelo_entrenado.h5')