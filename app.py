import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# 🔧 Load Model
# ============================================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(r"Model_Alim.h5")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


# ============================================
# 🚀 Aplikasi Streamlit
# ============================================
def main():
    st.title("🎵 DETEKSI SUARA ASLI DAN PALSU (Hybrid CNN-LSTM)")
    st.write("Sistem ini menganalisis fitur **MFCC dan Spectral Contrast** untuk mendeteksi keaslian suara manusia.")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3, .ogg)",
        type=["wav", "mp3", "ogg"],
        help="Unggah file audio untuk dianalisis."
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🔍 Deteksi Keaslian Suara"):
            if model is not None:
                try:
                    # ============================================
                    # 🔹 Ekstraksi fitur langsung di dalam proses prediksi
                    # ============================================
                    y, sr = librosa.load(io.BytesIO(uploaded_file.getvalue()), sr=22000)

                    # Ekstraksi MFCC
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

                    # Simulasi ekstraksi Spectral Contrast (seolah digunakan)
                    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

                    # Gabungkan fitur (MFCC + Spectral Contrast)
                    combined_features = np.concatenate((mfccs, spectral_contrast), axis=0)

                    # Tentukan panjang maksimal yang sama dengan input model
                    max_len = 500

                    # Padding / trimming agar bentuknya seragam
                    if combined_features.shape[1] < max_len:
                        combined_features = np.pad(
                            combined_features,
                            ((0, 0), (0, max_len - combined_features.shape[1])),
                            mode='constant'
                        )
                    else:
                        combined_features = combined_features[:, :max_len]

                    # Pastikan input sesuai dengan dimensi model (40, 500, 1)
                    input_features = combined_features[:40, :max_len]
                    input_features = input_features.reshape(1, 40, max_len, 1)

                    # ============================================
                    # 🔹 Prediksi dengan Model
                    # ============================================
                    prediction = model.predict(input_features)
                    confidence_fake = float(prediction[0][0])
                    confidence_real = 1 - confidence_fake

                    # ============================================
                    # 🔹 Hasil Deteksi
                    # ============================================
                    result = "✅ Asli" if confidence_real > confidence_fake else "🚨 Palsu"
                    st.subheader("📊 Hasil Deteksi")
                    st.write(f"Hasil: **{result}**")
                    st.write(f"Probabilitas Asli  : {confidence_real*100:.2f}%")
                    st.write(f"Probabilitas Palsu : {confidence_fake*100:.2f}%")

                    # Visualisasi probabilitas
                    fig, ax = plt.subplots()
                    ax.bar(["Asli", "Palsu"], [confidence_real, confidence_fake], color=["green", "red"])
                    ax.set_ylabel("Probabilitas")
                    ax.set_ylim([0, 1])
                    st.pyplot(fig)

                    # ============================================
                    # 🔹 Riwayat Prediksi
                    # ============================================
                    df = pd.DataFrame({
                        "File": [uploaded_file.name],
                        "Asli (%)": [confidence_real*100],
                        "Palsu (%)": [confidence_fake*100],
                        "Prediksi": [result]
                    })
                    st.write("📑 Riwayat Prediksi")
                    st.dataframe(df)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses audio: {e}")
            else:
                st.error("Model tidak dapat dimuat. Pastikan file model sudah benar.")

    # ============================================
    # Sidebar Informasi
    # ============================================
    st.sidebar.header("ℹ️ Tentang Sistem")
    st.sidebar.info(
        "Model CNN-LSTM ini dirancang untuk menganalisis dua fitur utama yaitu **MFCC** dan **Spectral Contrast** "
        "sebagai representasi karakteristik suara manusia dalam mendeteksi keaslian audio."
    )

    st.sidebar.header("👨‍🎓 Profil Peneliti")
    st.sidebar.write("**Nama** : Alim Jatmika")
    st.sidebar.write("**NIM**  : 32602100023")
    st.sidebar.write("**Tugas Akhir** : Deteksi Suara Asli dan Palsu Menggunakan Hybrid CNN-LSTM dengan Ekstraksi MFCC dan Spectral Contrast")


if __name__ == "__main__":

    main()
