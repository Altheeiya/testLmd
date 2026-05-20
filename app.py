import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Lateral Movement", layout="wide")

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

    return joblib.load('best_model.pkl')

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
    
    # C. Simple Graph Features (Metrik derajat node saja untuk efisiensi)
    G = nx.from_pandas_edgelist(df, 'source_ip', 'dest_ip')
    degree_dict = dict(G.degree())
    df['degree_centrality'] = df['source_ip'].map(degree_dict).fillna(0)
    
    # Mengisi kolom yang kosong agar sesuai dengan model training
    # (Pastikan fitur ini sesuai dengan yang ada di best_model)
    required_features = ['hour', 'is_business_hours', 'is_logon', 'is_failed_logon', 'is_process_create', 'degree_centrality']
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0
            
    return df[required_features]

def main():
    st.title("Deteksi Lateral Movement")
    st.write("Upload log Sysmon (CSV) untuk memproses fitur secara otomatis dan mendeteksi anomali.")

    try:
        model = load_model()
    except Exception:
        st.stop()

    uploaded_file = st.file_uploader("Upload Log Sysmon (CSV)", type=['csv'])
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Data Mentah (Preview):", raw_df.head())
        
        if st.button("Jalankan Full Process"):
            with st.spinner("Sedang memproses fitur..."):
                processed_df = process_data(raw_df)
                st.write("Fitur hasil proses (Input Model):", processed_df.head())
            
            with st.spinner("Melakukan prediksi..."):
                preds = model.predict(processed_df)
                raw_df['Prediksi'] = ["Lateral Movement" if p == 1 else "Normal" for p in preds]
                
                st.subheader("Hasil Akhir")
                st.dataframe(raw_df)
                st.download_button("Download Hasil", raw_df.to_csv().encode('utf-8'), "hasil.csv")

if __name__ == '__main__':
    main()