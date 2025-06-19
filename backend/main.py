from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import uvicorn
import cv2
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("son_model.keras")
IMG_SIZE = (128, 128)

# Sınıf isimlerini burada tutuyoruz
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

def prepare_image(file: UploadFile):
    img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _, img_array = prepare_image(file)
    prediction = model.predict(img_array)
    predicted_class_index = int(np.argmax(prediction, axis=1)[0])
    predicted_class_name = class_names[predicted_class_index]
    return {"prediction": predicted_class_name}

@app.post("/gradcam")
async def grad_cam(file: UploadFile = File(...)):
    original_img, img_array = prepare_image(file)
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_np = np.array(original_img)
    original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)
    output_path = "gradcam_result.jpg"
    cv2.imwrite(output_path, superimposed_img)
    return {"gradcam_url": f"/static/{output_path}"}

app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
