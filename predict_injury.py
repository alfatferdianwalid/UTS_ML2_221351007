import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="predict_injury.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Kemungkinan Cedera")
st.write("Masukkan informasi player untuk mendapatkan mengetahui kemungkinan cedera.")

# Form input pengguna
Player_Age = st.number_input("Player Age", min_value=0.0, max_value=100.0)
Player_Weight = st.number_input("Player Weight", min_value=0.0, max_value=100.0)
Player_Height = st.number_input("Player Height", min_value=0.0, max_value=1000.0)
Previous_Injury = st.number_input("Previous Injury", min_value=0.0, max_value=100.0)
Training_Intensity = st.number_input("Training intensity", min_value=0.0, max_value=100.0)
Recovery_Time = st.number_input("Recovery Time", min_value=0.0, max_value=14.0)

if st.button("Likehood of Injury"):
    # Preprocessing input
    input_data = np.array([[Player_Age, Player_Weight, Player_Height, Previous_Injury, Training_Intensity, Recovery_Time]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    likehood_injury = 'tidak cedera' if prediction == 0 else 'cedera'

    st.success(f"Likehood Injury: *{likehood_injury.upper()}*")