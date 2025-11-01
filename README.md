---
title: Plant Disease Detector
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

<div align="center">

# 🌱 Plant Disease Detector

**A deep learning web app to identify 38 different plant diseases from leaf images.**

This project uses a pre-trained EfficientNetB4 model, fine-tuned on a dataset of over 50,000 images, to classify plant diseases with high accuracy.


</div>

---

## 🚀 About The Project

This application was built to provide an accessible tool for farmers, gardeners, and researchers to quickly diagnose plant diseases. Early detection is crucial for managing crop health and preventing widespread damage.

The model is trained on the "Plant Village" dataset, which includes 38 distinct classes of diseases and healthy leaves from 14 different plant species.

### ✨ Features

* **High Accuracy:** Built on the powerful EfficientNetB4 architecture.
* **Wide Coverage:** Identifies 38 different disease categories.
* **User-Friendly:** Simple drag-and-drop interface powered by Gradio.
* **Fast & Free:** Deployed and hosted on Hugging Face Spaces.

### 🛠️ Built With

* [TensorFlow](https://www.tensorflow.org/)
* [Gradio](https://www.gradio.app/)
* [Hugging Face Spaces](https://huggingface.co/spaces)
* [NumPy](https://numpy.org/)
* [Pillow](https://python-pillow.org/)

---

## 🪴 How to Use

Using the app is simple:

1.  **Find an image** of a plant leaf you want to classify.
2.  **Drag and drop** the image into the input box on the app.
3.  **Click "Submit"** (or wait for the prediction).
4.  **View the results**, which will show the top 3 predicted diseases and their confidence scores.

---

## 📈 Model & Training

The model is a `tf.keras.Sequential` model that uses a pre-trained **EfficientNetB4** base, frozen for initial training, and then fine-tuned.

* **Dataset:** [New Plant Diseases Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1) (Mendeley Data)
* **Image Size:** `(160, 160, 3)`
* **Batch Size:** `32`
* **Total Epochs:** 16 (6 for the head, 10 for fine-tuning)
* **Final Test Accuracy:** `~98.5%` (You can update this with your actual number)

You can see the complete training process, data preprocessing, and model evaluation in the included Jupyter Notebook:
[**`notebooks/Plant_Disease_Model_Training.ipynb`**](./notebooks/Plant_Disease_Model_Training.ipynb)

---

## Acknowledgments

* **Dataset:** S. R. Dubey and A. S. Jalal, "Revised data of plant disease dataset," Mendeley Data, V1, doi: 10.17632/tywbtsjrjv.1
* **Tools:** [Hugging Face](https://huggingface.co/), [Gradio](https://www.gradio.app/), and [TensorFlow](https://www.tensorflow.org/).