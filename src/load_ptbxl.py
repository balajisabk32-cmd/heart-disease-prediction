import pandas as pd
import numpy as np
import os
import ast
import wfdb
import urllib.request
import ssl

os.makedirs('data/ptbxl/processed', exist_ok=True)

# ── 1. Load metadata ──────────────────────────────────────────────────────────
df_meta = pd.read_csv('data/ptbxl/ptbxl_database.csv', index_col='ecg_id')
print(f"Total records : {len(df_meta)}")

# ── 2. Load SCP statements (label definitions) ────────────────────────────────
# Download scp_statements.csv if not present
scp_path = 'data/ptbxl/scp_statements.csv'
if not os.path.exists(scp_path):
    print("Downloading scp_statements.csv...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    url = "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
    urllib.request.urlretrieve(url, scp_path)
    print("✅ scp_statements.csv downloaded")

scp_stmts = pd.read_csv(scp_path, index_col=0)
print(f"SCP statement categories:\n{scp_stmts['diagnostic_class'].value_counts()}\n")

# ── 3. Map scp_codes → binary label ──────────────────────────────────────────
# NORM = healthy (0), anything else with diagnostic confidence > 0 = disease (1)
def parse_scp(scp_str):
    """Parse string dict and return dominant label."""
    try:
        scp_dict = ast.literal_eval(scp_str)
    except:
        return 0

    # If NORM has highest confidence → healthy
    if 'NORM' in scp_dict:
        norm_conf = scp_dict['NORM']
        # Check if any pathology has higher confidence than NORM
        pathology_conf = {k: v for k, v in scp_dict.items() 
                         if k != 'NORM' and v > 0}
        if not pathology_conf:
            return 0  # pure normal
        max_path_conf = max(pathology_conf.values())
        if norm_conf >= max_path_conf:
            return 0  # NORM dominates
        return 1  # pathology dominates
    else:
        # No NORM code at all → disease
        return 1

df_meta['binary_label'] = df_meta['scp_codes'].apply(parse_scp)
print(f"Label distribution:")
print(df_meta['binary_label'].value_counts())
print(f"Disease rate: {df_meta['binary_label'].mean()*100:.1f}%\n")

# ── 4. Download a subset of actual ECG signal files ───────────────────────────
# We download 100 low-resolution records (100Hz, smaller files) per class
# Total: 200 records — enough for ECG encoder training on CPU

print("Downloading ECG signal files from PhysioNet...")
print("(This downloads ~200 records at 100Hz — small files)\n")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

base_url  = "https://physionet.org/files/ptb-xl/1.0.3/"
signal_dir = "data/ptbxl/signals/"
os.makedirs(signal_dir, exist_ok=True)

# Sample 100 from each class (stratified)
df_0 = df_meta[df_meta['binary_label'] == 0].sample(100, random_state=42)
df_1 = df_meta[df_meta['binary_label'] == 1].sample(100, random_state=42)
df_subset = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

print(f"Subset: {len(df_subset)} records (100 healthy + 100 disease)")

downloaded = 0
failed     = 0

for ecg_id, row in df_subset.iterrows():
    # filename_lr = low resolution (100Hz) path like "records100/00000/00001_lr"
    fname  = row['filename_lr']
    folder = os.path.dirname(fname)
    os.makedirs(os.path.join(signal_dir, folder), exist_ok=True)

    # Each WFDB record = 2 files: .dat and .hea
    for ext in ['.dat', '.hea']:
        local_path = os.path.join(signal_dir, fname + ext)
        if os.path.exists(local_path):
            continue
        remote_url = base_url + fname + ext
        try:
            urllib.request.urlretrieve(remote_url, local_path)
            downloaded += 1
        except Exception as e:
            failed += 1

    if (downloaded + failed) % 50 == 0 and downloaded > 0:
        print(f"  Progress: {downloaded} files downloaded, {failed} failed...")

print(f"\n✅ Download complete: {downloaded} files downloaded, {failed} failed")

# ── 5. Load signals + preprocess ──────────────────────────────────────────────
print("\nLoading and preprocessing ECG signals...")

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """Butterworth bandpass filter for ECG baseline wander removal."""
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

def load_ecg_record(fname, signal_dir):
    """Load one ECG record, return (1000, 12) array at 100Hz = 10 seconds."""
    try:
        record = wfdb.rdrecord(os.path.join(signal_dir, fname))
        sig    = record.p_signal  # shape: (samples, 12 leads)
        # Ensure exactly 1000 samples (10s @ 100Hz)
        if sig.shape[0] >= 1000:
            sig = sig[:1000, :]
        else:
            # Pad with zeros if shorter
            pad = np.zeros((1000 - sig.shape[0], 12))
            sig = np.vstack([sig, pad])
        # Bandpass filter
        sig = bandpass_filter(sig)
        # Z-score normalise per lead
        sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)
        return sig.astype(np.float32)
    except Exception as e:
        return None

signals = []
labels  = []
ecg_ids = []

for ecg_id, row in df_subset.iterrows():
    sig = load_ecg_record(row['filename_lr'], signal_dir)
    if sig is not None:
        signals.append(sig)
        labels.append(row['binary_label'])
        ecg_ids.append(ecg_id)

signals = np.array(signals)  # shape: (N, 1000, 12)
labels  = np.array(labels)

print(f"\nLoaded signals shape : {signals.shape}")
print(f"Labels shape         : {labels.shape}")
print(f"Label distribution   : {np.bincount(labels)}")

# ── 6. Save processed ECG data ────────────────────────────────────────────────
np.save('data/ptbxl/processed/ecg_signals.npy', signals)
np.save('data/ptbxl/processed/ecg_labels.npy',  labels)
np.save('data/ptbxl/processed/ecg_ids.npy',     np.array(ecg_ids))

print(f"\n✅ Saved ecg_signals.npy  — shape: {signals.shape}")
print(f"✅ Saved ecg_labels.npy   — shape: {labels.shape}")
print(f"✅ ECG pipeline complete — ready for ResNet1D training")