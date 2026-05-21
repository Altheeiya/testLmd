import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Sistem Deteksi Lateral Movement", layout="wide")

# ==============================================================================
# 1. HARDCODE DAFTAR FITUR (PENTING!)
# ==============================================================================
# Ganti list di bawah ini dengan 59 fitur persis sesuai output dari Kaggle Anda.
# Harus berurutan dan ejaannya sama persis (huruf besar/kecil).
FITUR_MODEL = [
    'hour', 'day_of_week', 'day_of_month', 'is_business_hours', 'is_weekend',
    'is_critical_logon', 'is_kerberos_event', 'is_ntlm_event', 'is_failed_logon',
    'is_explicit_cred', 'is_process_create', 'is_network_conn', 'is_process_access',
    'is_sysmon_event', 'source_in_degree', 'source_out_degree', 'source_pagerank',
    'source_betweenness', 'source_clustering', 'dest_in_degree', 'dest_out_degree',
    'dest_pagerank', 'dest_betweenness', 'dest_clustering', 'source_event_count',
    'source_unique_events', 'UtcTime_freq', 'ProcessGuid_freq', 'Image_freq',
    'FileVersion_freq', 'Description_freq', 'OriginalFileName_freq', 'CommandLine_freq',
    'CurrentDirectory_freq', 'LogonGuid_freq', 'LogonId_freq', 'Hashes_freq',
    'ParentProcessGuid_freq', 'ParentImage_freq', 'ParentCommandLine_freq',
    'SourceIp_freq', 'DestinationIp_freq', 'DestinationHostname_freq', 'ImageLoaded_freq',
    'SourceProcessGUID_freq', 'TargetProcessGUID_freq', 'TargetImage_freq',
    'CallTrace_freq', 'TargetFilename_freq', 'CreationUtcTime_freq', 'TargetObject_freq',
    'Details_freq', 'QueryName_freq', 'QueryResults_freq', 'TargetProcessGuid_freq',
    'StartAddress_freq', 'PreviousCreationUtcTime_freq', 'Hash_freq', 'SystemTime_freq'
]

# Mapping Label
LABEL_MAPPING = {
    0: "Normal",
    1: "Serangan (Tipe 1)",
    2: "Serangan Lateral Movement"
}

# ==============================================================================

@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

# Fungsi Pipeline Feature Engineering
def pipeline_feature_engineering(df_raw):
    df = df_raw.copy()
    
    # --- PROSES EKSTRAKSI (Simulasi) ---
    # Jika df_raw adalah data mentah, proses ekstraksi normalnya ada di sini.
    # Namun, untuk mencegah error saat presentasi, kita asumsikan file yang di-upload 
    # mungkin saja SUDAH berisi sebagian atau seluruh fitur tersebut.
    
    # 1. Temporal (Contoh ekstraksi jika fitur belum ada)
    if 'timestamp' in df.columns and 'hour' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        # df['is_business_hours'] = ... (tambahkan jika diperlukan)
        
    # 2. Graph Features (Sangat disederhanakan untuk demo Streamlit)
    if 'source_ip' in df.columns and 'dest_ip' in df.columns and 'source_out_degree' not in df.columns:
        try:
            G = nx.from_pandas_edgelist(df, 'source_ip', 'dest_ip', create_using=nx.DiGraph())
            out_deg = dict(G.out_degree())
            df['source_out_degree'] = df['source_ip'].map(out_deg).fillna(0)
            df['dest_in_degree'] = df['dest_ip'].map(dict(G.in_degree())).fillna(0)
            # Karena PageRank dan Betweenness butuh waktu komputasi lama, 
            # untuk demo Streamlit yang cepat, kita lewati atau beri nilai estimasi 0.
        except:
            pass

    # --- SINKRONISASI MATRIKS ---
    # Pastikan output akhir HANYA berisi 59 kolom yang ada di FITUR_MODEL.
    # Jika ada fitur yang tidak bisa diekstrak dari CSV mentah, beri nilai 0.
    X_matrix = pd.DataFrame()
    for col in FITUR_MODEL:
        if col in df.columns:
            # Pastikan tipe datanya float/int
            X_matrix[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            X_matrix[col] = 0.0  # Jika fitur tidak ada, paksa jadi 0.0
            
    # Pastikan urutan kolom sesuai dengan saat training
    X_matrix = X_matrix[FITUR_MODEL]
    return X_matrix

def main():
    st.title("Sistem Deteksi Anomali Lateral Movement")
    st.write("Aplikasi *Inference* SOC otomatis: Mendeteksi pergerakan lateral dari log Sysmon.")
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Gagal memuat file model pkl: {e}")
        return
        
    st.sidebar.header("Pusat Kendali Demo")
    uploaded_file = st.sidebar.file_uploader("Upload Raw Log Jaringan (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("1. Sampel Data Input")
        st.dataframe(df_raw.head(5))
        
        if st.sidebar.button("Jalankan Sistem Deteksi"):
            with st.spinner("Pipeline berjalan: Mengekstrak Fitur & Sinkronisasi Matriks..."):
                X_input = pipeline_feature_engineering(df_raw)
                
            with st.spinner("Model XGBoost memprediksi status keamanan..."):
                try:
                    # Proses Prediksi
                    predictions = model.predict(X_input)
                    df_raw['Prediksi'] = [LABEL_MAPPING.get(int(p), f"Anomali_{p}") for p in predictions]
                    
                    st.success("Analisis Deteksi Selesai!")
                    
                    # Ringkasan
                    st.subheader("2. Panel Ringkasan Keamanan SOC")
                    col1, col2, col3 = st.columns(3)
                    total_log = len(df_raw)
                    normal_log = (df_raw['Prediksi'] == "Normal").sum()
                    threat_log = total_log - normal_log
                    
                    col1.metric("Total Aktivitas Diperiksa", f"{total_log:,}")
                    col2.metric("Aktivitas Normal", f"{normal_log:,}")
                    col3.metric("Indikasi Intrusi", f"{threat_log:,}", delta=int(threat_log), delta_color="inverse")
                    
                    # Detail Tabel
                    st.subheader("3. Detail Data")
                    filter_opsi = st.selectbox("Saring Tampilan Tabel:", ["Hanya Tampilkan Serangan", "Tampilkan Semua"])
                    
                    if filter_opsi == "Hanya Tampilkan Serangan":
                        df_display = df_raw[df_raw['Prediksi'] != "Normal"]
                    else:
                        df_display = df_raw
                    
                    # Batasi tampilan agar tidak hang
                    df_final_display = df_display.head(1000)
                    
                    def color_row(val):
                        return 'background-color: #ffe6e6; color: black' if val != "Normal" else 'background-color: #e6ffe6; color: black'
                    
                    st.dataframe(df_final_display.style.map(color_row, subset=['Prediksi']), use_container_width=True)
                    
                    # Download
                    csv_file = df_raw.to_csv(index=False).encode('utf-8')
                    st.download_button("Unduh Laporan Lengkap (.CSV)", data=csv_file, file_name="laporan_deteksi.csv", mime="text/csv")
                
                except Exception as e:
                    st.error(f"Gagal saat melakukan prediksi: {e}")
                    st.info("Pastikan list 'FITUR_MODEL' di kode app.py sudah sama persis dengan fitur saat model dilatih.")

if __name__ == '__main__':
    main()