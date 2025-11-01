import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict(input_image: Image.Image):
    img = input_image.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    predictions = model.predict(preprocessed_img)
    
    scores = tf.nn.softmax(predictions[0])
    
    confidences = {class_names[i]: float(scores[i]) for i in range(len(class_names))}
    
    return confidences

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="🌱 Plant Disease Recognition",
    description="Upload an image of a plant leaf. The model will predict the disease and confidence.",
    examples=[
        
    ]
)
if __name__ == "__main__":
    iface.launch()