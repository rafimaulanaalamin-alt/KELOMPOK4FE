"""
Student Performance Prediction App
-----------------------------------
Aplikasi Streamlit untuk memprediksi status penempatan (placement) siswa
menggunakan model Random Forest Classifier.
"""

import streamlit as st
import pandas as pd
import joblib

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
)


# =========================================================
# LOAD ARTEFAK (MODEL, SCALER, FEATURE COLUMNS)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Memuat model, scaler, dan daftar kolom fitur.

    Mengembalikan tuple (model, scaler, feature_columns, error_message).
    Jika ada file yang gagal dimuat, error_message akan berisi pesan,
    selain itu None.
    """
    try:
        model = joblib.load("random_forest_model.pkl")
    except FileNotFoundError:
        return None, None, None, "File 'random_forest_model.pkl' tidak ditemukan."

    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        return None, None, None, (
            "File 'scaler.pkl' tidak ditemukan. "
            "Scaler wajib disimpan saat training agar preprocessing "
            "data baru konsisten dengan data training."
        )

    try:
        raw_feature_columns = joblib.load("feature_columns.pkl")
    except FileNotFoundError:
        return None, None, None, "File 'feature_columns.pkl' tidak ditemukan."

    # Buang target ('placement_status') dari daftar fitur bila ada,
    # untuk mencegah data leakage saat inference.
    feature_columns = [c for c in raw_feature_columns if c != "placement_status"]

    return model, scaler, feature_columns, None


model, scaler, feature_columns, load_error = load_artifacts()

if load_error:
    st.error(f"⚠️ Gagal memuat komponen model: {load_error}")
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.title("🎓 Student Performance Prediction")
st.caption("Demo Machine Learning menggunakan Random Forest Classifier")
st.divider()


# =========================================================
# FORM INPUT
# =========================================================
st.subheader("Data Siswa")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Jam Belajar per Hari", min_value=1.0, max_value=11.0, value=6.0, step=0.5
        )
        sleep_hours = st.number_input(
            "Jam Tidur per Hari", min_value=4.0, max_value=9.0, value=7.0, step=0.5
        )
        assignments_completed = st.number_input(
            "Tugas Diselesaikan", min_value=0, max_value=20, value=10, step=1
        )

    with col2:
        attendance = st.number_input(
            "Kehadiran (%)", min_value=40.0, max_value=100.0, value=70.0, step=1.0
        )
        internet_usage = st.number_input(
            "Penggunaan Internet (jam/hari)", min_value=1.0, max_value=11.0, value=6.0, step=0.5
        )
        previous_score = st.number_input(
            "Nilai Sebelumnya", min_value=35.0, max_value=95.0, value=65.0, step=1.0
        )

    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)


# =========================================================
# PREDIKSI
# =========================================================
def build_input_dataframe():
    """Menyusun input pengguna menjadi DataFrame sesuai urutan kolom training."""
    data = {
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "internet_usage": internet_usage,
        "assignments_completed": assignments_completed,
        "previous_score": previous_score,
    }

    df = pd.DataFrame([data])

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Kolom fitur berikut tidak tersedia pada input: {missing_cols}"
        )

    return df[feature_columns]


if submitted:
    try:
        with st.spinner("Menghitung prediksi..."):
            input_df = build_input_dataframe()
            input_scaled = pd.DataFrame(
                scaler.transform(input_df), columns=feature_columns
            )
            prediction = model.predict(input_scaled)[0]

            # Ambil probabilitas jika model mendukung, untuk konteks tambahan
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.success("✅ Status Prediksi: **Placed**")
        else:
            st.error("❌ Status Prediksi: **Not Placed**")

        if proba is not None:
            confidence = max(proba) * 100
            st.progress(min(int(confidence), 100))
            st.caption(f"Tingkat keyakinan model: {confidence:.1f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
