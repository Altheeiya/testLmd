import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Sistem Deteksi Lateral Movement", layout="wide")

# Mapping Label untuk Output Manusia
LABEL_MAPPING = {
    0: "Normal",
    1: "Serangan (Tipe 1)",
    2: "Serangan Lateral Movement"
}

# Memuat Model secara Aman dengan Caching
@st.cache_resource
def load_model():
    # Menunjuk langsung ke nama file pkl Anda di GitHub
    return joblib.load('best_model.pkl')

# Fungsi Otomatisasi Feature Engineering (Full Pipeline)
def pipeline_feature_engineering(df_raw, model_features):
    df = df_raw.copy()
    
    # 1. Ekstraksi Fitur Temporal (Waktu)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 2. Ekstraksi Fitur Event ID Biner (Sysmon)
    if 'event_id' in df.columns:
        df['is_logon'] = (df['event_id'] == 4624).astype(int)
        df['is_failed_logon'] = (df['event_id'] == 4625).astype(int)
        df['is_process_create'] = (df['event_id'] == 4688).astype(int)
        df['is_network_share'] = (df['event_id'] == 5140).astype(int)
    
    # 3. Pembentukan Graf Jaringan Terarah & Ekstraksi Fitur Topologi Jaringan
    # Memeriksa apakah kolom IP Sumber dan Tujuan tersedia
    if 'source_ip' in df.columns and 'dest_ip' in df.columns:
        # Membuat Graph dari struktur data relasional IP
        G = nx.from_pandas_edgelist(df, 'source_ip', 'dest_ip', create_using=nx.DiGraph())
        
        # Ekstraksi Metrik Derajat (In/Out Degree)
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        df['source_ip_in_degree'] = df['source_ip'].map(in_deg)
        df['dest_ip_in_degree'] = df['dest_ip'].map(in_deg)
        df['source_ip_out_degree'] = df['source_ip'].map(out_deg)
        df['dest_ip_out_degree'] = df['dest_ip'].map(out_deg)
        
        # Algoritma PageRank
        try:
            pr = nx.pagerank(G, alpha=0.85)
            df['source_ip_pagerank'] = df['source_ip'].map(pr)
            df['dest_ip_pagerank'] = df['dest_ip'].map(pr)
        except:
            df['source_ip_pagerank'] = 0.0
            df['dest_ip_pagerank'] = 0.0
            
        # Algoritma Jaringan Tambahan (Frekuensi Aktivitas)
        df['source_event_count'] = df['source_ip'].map(df['source_ip'].value_counts())
    
    # 4. Sinkronisasi dengan Struktur Otak Model Latih (59 Fitur)
    # Membuat DataFrame kosong dengan kolom yang dibutuhkan model latih
    X_matrix = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in df.columns:
            X_matrix[col] = df[col]
        else:
            X_matrix[col] = 0 # Nilai substitusi default jika fitur tidak muncul pada log demo
            
    # Membersihkan nilai kosong/tak terhingga
    X_matrix = X_matrix.fillna(0).replace([np.inf, -np.inf], 0)
    return X_matrix

def main():
    st.title("Sistem Deteksi Anomali Lateral Movement Berbasis Algoritma Graf")
    st.write("Aplikasi *Inference* SOC otomatis: Mendeteksi pergerakan lateral peretas dari log mentah Windows Sysmon.")
    
    # Load Model Penolak Masalah Path
    try:
        model = load_model()
        # Membaca daftar fitur asli dari objek XGBoost
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
    except Exception as e:
        st.error(f"Gagal memuat file model pkl: {e}")
        return
        
    # Sidebar menu upload data mentah
    st.sidebar.header("Pusat Kendali Demo")
    uploaded_file = st.sidebar.file_uploader("Upload Raw Log Jaringan (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Membaca file input mentah
        df_raw = pd.read_csv(uploaded_file)
        
        st.subheader("1. Sampel Log Mentah Masuk (Raw Data)")
        st.dataframe(df_raw.head(5))
        
        if st.sidebar.button("Jalankan Sistem Deteksi"):
            with st.spinner("Pipeline berjalan: Mengekstrak Fitur Graf & Waktu..."):
                # Menjalankan pemrosesan otomatis
                X_input = pipeline_feature_engineering(df_raw, model_features)
                
            with st.spinner("Model XGBoost memprediksi status keamanan jaringan..."):
                # Proses Klasifikasi Inferensi
                predictions = model.predict(X_input)
                
                # Konversi angka prediksi menjadi teks label aslinya
                df_raw['Prediksi'] = [LABEL_MAPPING.get(int(p), f"Anomali_{p}") for p in predictions]
                
                st.success("Analisis Deteksi Selesai!")
                
                # Tampilan Dashboard Ringkasan Metrik
                st.subheader("2. Panel Ringkasan Keamanan SOC")
                col1, col2, col3 = st.columns(3)
                total_log = len(df_raw)
                normal_log = (df_raw['Prediksi'] == "Normal").sum()
                threat_log = total_log - normal_log
                
                col1.metric("Total Aktivitas Diperiksa", f"{total_log:,}")
                col2.metric("Aktivitas Normal Aman", f"{normal_log:,}")
                col3.metric("Indikasi Intrusi/Anomali", f"{threat_log:,}", delta=int(threat_log), delta_color="inverse")
                
                # Manajemen Visualisasi Data (Mencegah Websocket Error/Disconnect)
                st.subheader("3. Detail Data Terdampak Serangan")
                filter_opsi = st.selectbox("Saring Tampilan Tabel:", ["Tampilkan Semua", "Hanya Tampilkan Serangan"])
                
                if filter_opsi == "Hanya Tampilkan Serangan":
                    df_display = df_raw[df_raw['Prediksi'] != "Normal"]
                else:
                    df_display = df_raw
                
                # Memotong data yang tampil di monitor browser maksimal 1.000 baris agar ringan
                MAX_DISPLAY = 1000
                df_final_display = df_display.head(MAX_DISPLAY)
                st.write(f"Menampilkan {len(df_final_display)} data teratas di layar monitor:")
                
                # Fungsi pewarnaan baris tabel Pandas Styler (Kompatibel Pandas 3.0)
                def color_row(val):
                    return 'background-color: #ffe6e6; color: black' if val != "Normal" else 'background-color: #e6ffe6; color: black'
                
                st.dataframe(df_final_display.style.map(color_row, subset=['Prediksi']), use_container_width=True)
                
                # Tombol Download data utuh (tidak terpotong 1.000 baris)
                st.subheader("4. Pelaporan")
                csv_file = df_raw.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Unduh Laporan Deteksi Lengkap (.CSV)",
                    data=csv_file,
                    file_name="laporan_analisis_lateral_movement.csv",
                    mime="text/csv"
                )
    else:
        st.info("Sistem Siaga (Idle). Silakan unggah sampel log mentah jaringan komputer melalui panel menu kiri untuk memulai deteksi.")

if __name__ == '__main__':
    main()