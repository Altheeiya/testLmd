import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Lateral Movement", layout="wide")

REQUIRED_RAW_COLUMNS = ["timestamp", "event_id", "source_ip", "dest_ip"]
CHUNK_SIZE = 50000

# Muat Model
@st.cache_resource
def load_model():
    try:
        import joblib
    except ModuleNotFoundError as exc:
        st.error(
            "Dependency `joblib` belum terpasang di environment deployment. "
            "Pastikan `requirements.txt` sudah dipakai saat deploy."
        )
        raise exc

    model_path = Path(__file__).resolve().parent / 'best_model.pkl'
    return joblib.load(model_path)

# 1. LOGIKA FULL PREPROCESSING
def process_data(df):
    # A. Temporal Features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
    
    # B. EventID Features (Hardcode mapping)
    event_mapping = {4624: 'is_logon', 4625: 'is_failed_logon', 4688: 'is_process_create'}
    for eid, name in event_mapping.items():
        df[name] = (df['event_id'] == eid).astype(int)
    
    # C. Simple Graph Features (vectorized agar lebih ringan untuk file besar)
    source_counts = df['source_ip'].value_counts(dropna=False)
    dest_counts = df['dest_ip'].value_counts(dropna=False)
    df['degree_centrality'] = (
        df['source_ip'].map(source_counts).fillna(0)
        + df['dest_ip'].map(dest_counts).fillna(0)
    ).astype(int)
    
    # Mengisi kolom yang kosong agar sesuai dengan model training
    # (Pastikan fitur ini sesuai dengan yang ada di best_model)
    required_features = ['hour', 'is_business_hours', 'is_logon', 'is_failed_logon', 'is_process_create', 'degree_centrality']
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0
            
    return df[required_features]


def read_preview(uploaded_file):
    uploaded_file.seek(0)
    preview_df = pd.read_csv(uploaded_file, usecols=lambda column: column in REQUIRED_RAW_COLUMNS, nrows=5)
    uploaded_file.seek(0)
    return preview_df


def process_uploaded_file(uploaded_file):
    uploaded_bytes = uploaded_file.getvalue()
    csv_buffer = BytesIO(uploaded_bytes)
    model = load_model()

    processed_parts = []
    raw_parts = []

    for chunk in pd.read_csv(csv_buffer, usecols=lambda column: column in REQUIRED_RAW_COLUMNS, chunksize=CHUNK_SIZE):
        missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in chunk.columns]
        if missing_columns:
            raise ValueError(f"Kolom wajib belum ada di CSV: {', '.join(missing_columns)}")

        processed_chunk = process_data(chunk.copy())
        predictions = model.predict(processed_chunk)

        chunk = chunk.copy()
        chunk['Prediksi'] = ["Lateral Movement" if prediction == 1 else "Normal" for prediction in predictions]
        processed_parts.append(chunk)
        raw_parts.append(processed_chunk)

    if not processed_parts:
        raise ValueError("CSV kosong atau tidak dapat diproses.")

    return pd.concat(processed_parts, ignore_index=True), pd.concat(raw_parts, ignore_index=True)

def main():
    st.title("Deteksi Lateral Movement")
    st.write("Upload log Sysmon (CSV) untuk memproses fitur secara otomatis dan mendeteksi anomali.")
    st.caption("Aplikasi ini hanya memproses kolom yang diperlukan: timestamp, event_id, source_ip, dan dest_ip.")

    uploaded_file = st.file_uploader("Upload Log Sysmon (CSV)", type=['csv'])
    
    if uploaded_file:
        if uploaded_file.size and uploaded_file.size > 150 * 1024 * 1024:
            st.error("File terlalu besar untuk diproses di Streamlit Cloud. Pecah CSV menjadi beberapa bagian yang lebih kecil dari 150 MB.")
            st.stop()

        try:
            raw_df = read_preview(uploaded_file)
        except ValueError:
            st.error("CSV harus memiliki kolom: timestamp, event_id, source_ip, dan dest_ip.")
            st.stop()

        missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in raw_df.columns]
        if missing_columns:
            st.error(f"Kolom wajib belum ada di CSV: {', '.join(missing_columns)}")
            st.stop()

        st.write("Data Mentah (Preview):", raw_df.head())
        
        if st.button("Jalankan Full Process"):
            try:
                with st.spinner("Sedang memproses file besar per bagian..."):
                    result_df, processed_df = process_uploaded_file(uploaded_file)

                st.write("Fitur hasil proses (Input Model):", processed_df.head())
                st.subheader("Hasil Akhir")
                st.dataframe(result_df)
                st.download_button("Download Hasil", result_df.to_csv(index=False).encode('utf-8'), "hasil.csv")
            except Exception as exc:
                st.error("Terjadi kesalahan saat memproses file besar.")
                st.exception(exc)

if __name__ == '__main__':
    main()