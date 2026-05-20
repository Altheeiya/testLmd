import streamlit as st
import pandas as pd
import joblib
import json
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

# Load expected feature names used by the model
FEATURE_NAMES_PATH = Path(__file__).resolve().parent / 'feature_names.json'
if FEATURE_NAMES_PATH.exists():
    with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as fh:
        EXPECTED_FEATURES = json.load(fh)
else:
    EXPECTED_FEATURES = None

# Load label mapping if present
LABEL_MAPPING_PATH = Path(__file__).resolve().parent / 'label_mapping.json'
if LABEL_MAPPING_PATH.exists():
    with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as fh:
        LABEL_MAP = json.load(fh)
else:
    LABEL_MAP = None

# Derive a human-readable label map: if LABEL_MAP values are just numeric/identity,
# convert to Normal (0) / Lateral Movement (non-zero). Otherwise use LABEL_MAP as-is.
HUMAN_LABEL_MAP = None
if LABEL_MAP:
    try:
        values_are_numeric = all(str(v).strip().isdigit() for v in LABEL_MAP.values())
        keys_equal_values = all(str(k) == str(v) for k, v in LABEL_MAP.items())
    except Exception:
        values_are_numeric = False
        keys_equal_values = False

    if values_are_numeric or keys_equal_values:
        HUMAN_LABEL_MAP = {str(k): ('Normal' if str(k) in ('0', '0.0') else 'Lateral Movement') for k in LABEL_MAP.keys()}
    else:
        # assume LABEL_MAP already maps to human-readable labels
        HUMAN_LABEL_MAP = {str(k): str(v) for k, v in LABEL_MAP.items()}

else:
    HUMAN_LABEL_MAP = None

# Muat Model
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / 'best_model.pkl'
    return joblib.load(model_path)


def normalize_input_columns(df):
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "").replace("-", "_")

    cols = list(df.columns)
    col_norm = {col: _norm(col) for col in cols}

    result = df.copy()

    for canonical_name, aliases in RAW_COLUMN_ALIASES.items():
        alias_norms = set(_norm(a) for a in aliases)
        # find all columns that match any alias for this canonical
        matched = [col for col, n in col_norm.items() if n in alias_norms]
        if not matched:
            continue

        # If canonical already exists, include it first so it takes precedence
        cols_to_merge = []
        if canonical_name in result.columns:
            cols_to_merge.append(canonical_name)
        # add other matched columns (excluding canonical if present)
        cols_to_merge.extend([c for c in matched if c != canonical_name])

        # Create canonical column as first non-null across matched columns
        try:
            result[canonical_name] = result[cols_to_merge].bfill(axis=1).iloc[:, 0]
        except Exception:
            # Fallback: if bfill fails (e.g., single column), just take first matched
            result[canonical_name] = result[cols_to_merge[0]]

        # Drop the original matched columns except the canonical
        for c in matched:
            if c != canonical_name and c in result.columns:
                result.drop(columns=c, inplace=True)

    # Ensure column names are unique now
    result = result.loc[:, ~result.columns.duplicated()]
    return result

# 1. LOGIKA FULL PREPROCESSING
def process_data(df):
    df = normalize_input_columns(df.copy())

    missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]

    # Heuristik: coba deteksi kolom yang mirip jika ada (mis-named exports)
    if 'timestamp' in missing_columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'created', 'eventtime'])]
        if candidates:
            df['timestamp'] = df[candidates[0]]
            missing_columns.remove('timestamp')

    if 'source_ip' in missing_columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['src', 'source', 'client', 'ip', 'address'])]
        if candidates:
            df['source_ip'] = df[candidates[0]]
            missing_columns.remove('source_ip')

    if 'dest_ip' in missing_columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['dst', 'dest', 'destination', 'target', 'ip', 'address'])]
        # prefer candidates that are not the same as source_ip
        candidates = [c for c in candidates if c not in (['source_ip'] + list(df.columns)) or True]
        if candidates:
            df['dest_ip'] = df[candidates[0]]
            missing_columns.remove('dest_ip')

    if missing_columns:
        raise ValueError(f"Kolom wajib belum ada di CSV: {', '.join(missing_columns)}")

    # A. Temporal Features
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as exc:
        raise ValueError(f"Gagal mengurai kolom timestamp: {exc}. Kolom tersedia: {', '.join(df.columns)}")
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
    base_features = ['hour', 'is_business_hours', 'is_logon', 'is_failed_logon', 'is_process_create', 'degree_centrality']
    for feat in base_features:
        if feat not in df.columns:
            df[feat] = 0

    # If we have a feature_names.json, ensure final DF matches that ordering and fills missing features with 0
    if EXPECTED_FEATURES:
        for feat in EXPECTED_FEATURES:
            if feat not in df.columns:
                df[feat] = 0
            # coerce to numeric, fill NaN with 0
            df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0).astype(float)
        return df[EXPECTED_FEATURES]

    # Fallback: return base features if no feature list provided
    return df[base_features]


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
    for chunk in pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE):
        normalized_chunk = normalize_input_columns(chunk)
        processed_chunk = process_data(normalized_chunk)
        predictions = model.predict(processed_chunk)

        output_chunk = normalized_chunk.copy()
        # store raw numeric prediction for later mapping/triage
        output_chunk['Prediksi_raw'] = predictions

        result_parts.append(output_chunk)
        feature_parts.append(processed_chunk)

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

                # Map raw predictions to human labels using label_map if available
                if HUMAN_LABEL_MAP:
                    # Map raw numeric predictions to human-readable labels
                    result_df['Prediksi'] = result_df['Prediksi_raw'].astype(str).map(lambda x: HUMAN_LABEL_MAP.get(x, f"Class_{x}"))
                else:
                    result_df['Prediksi'] = result_df['Prediksi_raw'].apply(lambda p: f"Class_{p}")

                # Advanced: manual selector hidden by default to avoid confusion
                selected_anom = None
                show_manual = st.checkbox("Advanced: Manual anomaly selector (tampilkan pilihan kelas mentah)", value=False)
                if show_manual:
                    unique_raw = sorted(result_df['Prediksi_raw'].astype(str).unique().tolist())
                    default_anom = [str(x) for x in unique_raw if int(float(x)) != 0]
                    selected_anom = st.multiselect("Pilih kelas yang dianggap Lateral Movement (raw)", options=unique_raw, default=default_anom)

                display_df = result_df.copy()
                if selected_anom:
                    # selected_anom are strings from multiselect; ensure types
                    selected_set = set(int(x) for x in selected_anom)
                    display_df = display_df[display_df['Prediksi_raw'].isin(selected_set)]

                # Totals summary
                total_rows = len(result_df)
                # count anomalies according to selected raw classes
                if selected_anom:
                    anomaly_count = int(result_df['Prediksi_raw'].isin(selected_set).sum())
                else:
                    # default: any non-zero raw label
                    anomaly_count = int((result_df['Prediksi_raw'] != 0).sum())
                normal_count = total_rows - anomaly_count

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Baris", f"{total_rows}")
                c2.metric("Lateral Movement (dipilih)", f"{anomaly_count}")
                c3.metric("Normal", f"{normal_count}")

                st.subheader("Hasil Akhir")
                st.dataframe(display_df)
                st.download_button("Download Hasil", display_df.to_csv(index=False).encode('utf-8'), "hasil.csv")

                # IP summary for triage
                lateral_counts = result_df[result_df['Prediksi'] == 'Lateral Movement']['source_ip'].value_counts()
                if not lateral_counts.empty:
                    st.subheader('Ringkasan IP Sumber (Top 10)')
                    st.table(lateral_counts.head(10).rename_axis('source_ip').reset_index(name='count'))

                # Diagnostics for debugging model/input issues
                with st.expander("Diagnostics (developer)"):
                    model_info = load_model()
                    st.write("Model classes:", getattr(model_info, 'classes_', None))
                    st.write("Model n_features_in_:", getattr(model_info, 'n_features_in_', None))
                    st.write("Prediksi_raw value counts:")
                    st.write(result_df['Prediksi_raw'].value_counts())
                    st.write("Sample processed features (first 5 rows):")
                    st.dataframe(processed_df.head())
                    # show which features are entirely zero (possible preprocessing bug)
                    try:
                        zero_feats = (processed_df.sum(axis=0) == 0)
                        if zero_feats.any():
                            st.write("Fitur dengan jumlah 0 sepanjang dataset:", zero_feats[zero_feats].index.tolist())
                        else:
                            st.write("Tidak ada fitur yang semuanya nol.")
                    except Exception as e:
                        st.write("Gagal menghitung ringkasan fitur:", e)

            except Exception as exc:
                st.error("Terjadi kesalahan saat memproses file besar.")
                st.exception(exc)

if __name__ == '__main__':
    main()