
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import openai

openai.api_key = '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'

app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    model = tf.keras.models.load_model('/Users/jarvis/pymycod/tensorflow_AI/trained_models/lr_model_chest_xray.h5')
    img = cv2.imread(file)
    img = cv2.resize(img, (300, 300))
    img = np.expand_dims(img, axis = 0)
    ans = model.predict(img)
    if ans <=2.8:
        ans = "pnuemonia"
        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"give me possible treatments for {ans} and also give me all the required home remedies with precautions i can take",
        max_tokens=200
  )
        return response.choices[0].text.strip()
    else:
        ans = "normal"
        return "you are absolutely fine"
    
    




@app.get("/predict")
async def get_prediction():
    prediction = upload_file()
    return {"prediction": prediction}





