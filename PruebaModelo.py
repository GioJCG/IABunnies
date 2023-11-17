import cv2
import numpy as np
from tensorflow.keras.models import load_model

modelo_entrenado = load_model('model/modelo_entrenado.h5')

listaPersonas = [
    'Daniela',
    'Gerardo',
    'Jazael',
    'Melissa',
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:

        face_roi = gray[y:y + h, x:x + w]

        face_roi_resized = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_AREA)

        face_roi_normalized = face_roi_resized / 255.0

        face_input = np.expand_dims(face_roi_normalized, axis=0)

        prediction = modelo_entrenado.predict(face_input)
        predicted_person = listaPersonas[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Predicción de Rostros', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()