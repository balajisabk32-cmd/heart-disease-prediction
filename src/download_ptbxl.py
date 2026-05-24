import os
import urllib.request
import zipfile

os.makedirs('data/ptbxl', exist_ok=True)

print("Downloading PTB-XL metadata...")

# PTB-XL metadata CSV (labels for all 21,799 records)
meta_url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"

# Use no SSL verification workaround for Windows
import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

try:
    urllib.request.urlretrieve(meta_url, 'data/ptbxl/ptbxl_database.csv')
    print("✅ Metadata downloaded")
except Exception as e:
    print(f"Direct download failed: {e}")
    print("Using fallback method...")

    import subprocess
    result = subprocess.run([
        'curl', '-k', '-o', 'data/ptbxl/ptbxl_database.csv', meta_url
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Metadata downloaded via curl")
    else:
        print("❌ curl also failed. Will use synthetic ECG data instead.")
        print("Run: python src/generate_synthetic_ecg.py")