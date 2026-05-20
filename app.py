import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Lateral Movement", layout="wide")

REQUIRED_RAW_COLUMNS = ["timestamp", "event_id", "source_ip", "dest_ip"]
RAW_COLUMN_ALIASES = {
    "timestamp": ["utctime", "systemtime", "timecreated", "timestamp", "time",
                  "datetime", "eventtime", "event_time", "creationutctime"],
    "event_id": ["event_id", "eventid", "event id", "eventcode", "event_code"],
    "source_ip": ["source_ip", "src_ip", "sourceip", "source ip", "source_address"],
    "dest_ip": ["dest_ip", "dst_ip", "destination_ip", "destinationip", "dest ip", "target_ip"],
}
CHUNK_SIZE = 50000
MAX_UPLOAD_SIZE = 150 * 1024 * 1024  # 150 MB

# Muat Model
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / 'best_model.pkl'
    return joblib.load(model_path)


def normalize_input_columns(df):
    normalized_lookup = {
        str(column).strip().lower().replace(" ", "").replace("-", "_"): column
        for column in df.columns
    }
    rename_map = {}

    for canonical_name, aliases in RAW_COLUMN_ALIASES.items():
        for alias in aliases:
            lookup_key = alias.strip().lower().replace(" ", "").replace("-", "_")
            if lookup_key in normalized_lookup:
                original_name = normalized_lookup[lookup_key]
                if original_name != canonical_name:
                    rename_map[original_name] = canonical_name
                break

    return df.rename(columns=rename_map)

# 1. LOGIKA FULL PREPROCESSING
def process_data(df):
    df = normalize_input_columns(df.copy())

    missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib belum ada di CSV: {', '.join(missing_columns)}")

    # A. Temporal Features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
    
    # B. EventID Features (Hardcode mapping)
    event_mapping = {4624: 'is_logon', 4625: 'is_failed_logon', 4688: 'is_process_create'}
    for eid, name in event_mapping.items():
        df[name] = (df['event_id'] == eid).astype(int)
    
    # C. Simple Graph Features (lebih ringan untuk file besar)
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
    preview_df = pd.read_csv(uploaded_file, nrows=5)
    uploaded_file.seek(0)
    return normalize_input_columns(preview_df)


def process_uploaded_file(uploaded_file):
    model = load_model()

    result_parts = []
    feature_parts = []

    uploaded_file.seek(0)
    chunk_index = 0
    for chunk in pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE):
        normalized_chunk = normalize_input_columns(chunk)
        processed_chunk = process_data(normalized_chunk)
        predictions = model.predict(processed_chunk)

        output_chunk = normalized_chunk.copy()
        output_chunk['Prediksi'] = ["Lateral Movement" if pred == 1 else "Normal" for pred in predictions]

        result_parts.append(output_chunk)
        feature_parts.append(processed_chunk)
        chunk_index += 1

    if not result_parts:
        raise ValueError("CSV kosong atau tidak dapat diproses.")

    return pd.concat(result_parts, ignore_index=True), pd.concat(feature_parts, ignore_index=True)

def main():
    st.title("Full Process: Deteksi Lateral Movement")
    st.write("Upload log Sysmon (CSV) untuk memproses fitur secara otomatis dan mendeteksi anomali.")

    st.caption("Untuk file besar, aplikasi memproses data per-bagian dan hanya membutuhkan kolom yang bisa dipetakan ke timestamp, event_id, source_ip, dan dest_ip.")

    uploaded_file = st.file_uploader("Upload Log Sysmon (CSV)", type=['csv'])
    
    if uploaded_file:
        if uploaded_file.size and uploaded_file.size > MAX_UPLOAD_SIZE:
            st.error("File terlalu besar untuk Streamlit Cloud. Pecah CSV menjadi beberapa bagian yang lebih kecil dari 150 MB.")
            st.stop()

        try:
            raw_df = read_preview(uploaded_file)
        except Exception as exc:
            st.error("CSV tidak bisa dibaca.")
            st.exception(exc)
            st.stop()

        st.write("Data Mentah (Preview):", raw_df.head())
        
        if st.button("Jalankan Full Process"):
            try:
                progress = st.progress(0)
                with st.spinner("Sedang memproses file per bagian..."):
                    uploaded_file.seek(0)
                    result_df, processed_df = process_uploaded_file(uploaded_file)

                # persist results in session state
                st.session_state['result_df'] = result_df
                st.session_state['processed_df'] = processed_df

                st.write("Fitur hasil proses (Input Model):", processed_df.head())

                # quick filter for lateral movements
                show_only_lateral = st.checkbox("Tampilkan hanya Lateral Movement")
                display_df = result_df
                if show_only_lateral:
                    display_df = result_df[result_df['Prediksi'] == 'Lateral Movement']

                st.subheader("Hasil Akhir")
                st.dataframe(display_df)
                st.download_button("Download Hasil", display_df.to_csv(index=False).encode('utf-8'), "hasil.csv")

                # IP summary for triage
                lateral_counts = result_df[result_df['Prediksi'] == 'Lateral Movement']['source_ip'].value_counts()
                if not lateral_counts.empty:
                    st.subheader('Ringkasan IP Sumber (Top 10)')
                    st.table(lateral_counts.head(10).rename_axis('source_ip').reset_index(name='count'))

            except Exception as exc:
                st.error("Terjadi kesalahan saat memproses file besar.")
                st.exception(exc)

if __name__ == '__main__':
    main()