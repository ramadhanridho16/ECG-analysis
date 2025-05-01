import streamlit as st
import numpy as np
import joblib
from PIL import Image
import io

# Cache loading model supaya tidak load ulang setiap waktu
@st.cache_resource
def load_model():
    model = joblib.load('../model/xgb_model_ekg.joblib')
    return model

model = load_model()

st.title("Prediksi Detak Jantung EKG")

uploaded_file = st.file_uploader("Upload Gambar EKG", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar EKG Diupload", use_column_width=True)

    # Convert gambar ke array
    image_array = np.array(image.convert('L'))  # Convert ke grayscale
    image_flat = image_array.flatten()          # Di-flatten (ubah jadi 1D array)

    # Normalisasi sederhana
    image_flat = image_flat[:30]

    # Prediksi
    prediction = model.predict([image_flat])

    st.success(f"Prediksi Kategori: {prediction[0]}")

    # Setelah prediksi label
    # predicted_label = model.predict([prediction][0])

    # st.success(f'Prediksi Kategori: {predicted_label}')

    # Menampilkan deskripsi berdasarkan label prediksi
    label_descriptions = {
        0: "Normal: Sinyal EKG normal, tidak menunjukkan kelainan.",
        1: "Aritmia Ringan: Terdapat sedikit ketidakaturan pada detak jantung.",
        2: "Aritmia Sedang: Gangguan irama jantung terlihat, perlu perhatian medis.",
        3: "Aritmia Berat: Gangguan serius, perlu intervensi medis segera.",
        4: "Noise/Data Tidak Valid: Sinyal terganggu atau kesalahan perekaman."
    }

    st.info(label_descriptions.get(prediction[0], "Kategori tidak diketahui."))

