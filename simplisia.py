import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Daftar label kelas
labels = ['gingseng','jahe', 'kayu manis', 'kunyit', 'lengkuas', 'mahkota dewa', 'meniran', 'pace', 'sambiloto','sereh']

# Judul aplikasi
st.title("Image Classification with Custom Transfer Learning Model")

# Input untuk mengunggah gambar
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Variabel untuk menentukan apakah tombol prediksi ditekan
predict_button_pressed = False

# Tambahkan tombol untuk prediksi
if st.button('Predict') and uploaded_file is not None:
    predict_button_pressed = True

# Cek apakah tombol prediksi ditekan dan gambar diunggah
if predict_button_pressed and uploaded_file is not None:
    # Membaca gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Mengubah ukuran gambar menjadi (224, 224)
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    # Preprocess gambar
    image_array = np.expand_dims(image_array, axis=0)  # Menambahkan batch dimension
    image_array = preprocess_input(image_array)
    
    # Memuat model yang telah dilatih
    model = load_model('transfer_learning.keras')
    
    # Melakukan prediksi
    predictions = model.predict(image_array)
    
    # Menghitung argmax untuk mendapatkan kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Mendapatkan label kelas yang sesuai
    predicted_label = labels[predicted_class]
    
    # Menampilkan hasil prediksi dalam tampilan mirip card
    st.subheader("Prediction Result:")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("Predicted Class:")
    with col2:
        st.write(predicted_label)
    
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{labels[i]}: {prob * 100:.2f}%")
