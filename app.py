from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('model/modelo_entrenado.h5')

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        image_array = cv2.resize(image_array, (150, 150))
        image_array = image_array / 255.0
        prediction = model.predict(np.array([image_array]).reshape(-1, 150, 150, 1))
        predicted_label = np.argmax(prediction)
        listaPersonas = ['Daniela', 'Gerardo', 'Jazael']
        predicted_name = listaPersonas[predicted_label]
        predicted_label = int(predicted_label)
        predicted_name = str(predicted_name)

        return JSONResponse(content={'predicted_label': predicted_label, 'predicted_name': predicted_name})
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)
