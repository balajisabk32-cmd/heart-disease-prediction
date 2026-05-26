"""
Mock Patient Data Generator + Model Inference
Tests the Kardia multimodal fusion model on realistic synthetic patients.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import sys

# ── Add src to path ───────────────────────────────────────────────────────────
sys.path.append('src')
from resnet1d import ResNet1D

# ── 20 Mock Patients (realistic clinical profiles) ────────────────────────────
# Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
#          oldpeak, slope, ca, thal
# sex: 1=Male, 0=Female
# cp: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
# thal: 1=normal, 2=fixed defect, 3=reversable defect
# target: 0=healthy, 1=disease (ground truth for accuracy check)

mock_data = {
    'name': [
        'Arjun M, 55M',    'Priya S, 62F',    'Ravi K, 48M',     'Meena T, 70F',
        'Suresh P, 45M',   'Lakshmi R, 58F',  'Karthik N, 63M',  'Anitha D, 52F',
        'Vijay G, 67M',    'Deepa V, 41F',    'Rajesh B, 72M',   'Kavitha L, 55F',
        'Arun S, 38M',     'Sumathi C, 65F',  'Mohan R, 59M',    'Saranya K, 44F',
        'Ganesh P, 76M',   'Revathi N, 50F',  'Senthil A, 61M',  'Padma V, 68F'
    ],
    'age':      [55, 62, 48, 70, 45, 58, 63, 52, 67, 41, 72, 55, 38, 65, 59, 44, 76, 50, 61, 68],
    'sex':      [ 1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
    'cp':       [ 0,  2,  1,  0,  3,  1,  0,  2,  0,  2,  0,  1,  2,  0,  0,  1,  0,  2,  0,  1],
    'trestbps': [145,130,120,160,110,135,150,125,165,115,170,130,118,155,148,120,175,128,155,145],
    'chol':     [233,250,195,280,200,215,260,190,300,175,320,210,185,270,245,195,340,205,265,235],
    'fbs':      [ 1,  0,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1],
    'restecg':  [ 0,  1,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  0,  0],
    'thalach':  [150,162,187,130,178,165,140,172,120,185,110,158,195,135,145,175,105,162,138,148],
    'exang':    [ 0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  1,  0],
    'oldpeak':  [2.3,0.6,0.0,3.5,0.0,0.8,2.8,0.4,4.0,0.0,5.0,1.2,0.0,3.2,2.0,0.2,4.5,0.6,2.5,1.8],
    'slope':    [ 0,  2,  2,  0,  2,  2,  1,  2,  0,  2,  0,  1,  2,  0,  1,  2,  0,  2,  0,  1],
    'ca':       [ 0,  0,  0,  2,  0,  0,  2,  0,  3,  0,  3,  1,  0,  2,  1,  0,  3,  0,  2,  1],
    'thal':     [ 1,  2,  2,  3,  2,  2,  3,  2,  3,  2,  3,  2,  2,  3,  3,  2,  3,  2,  3,  2],
    # Ground truth (based on clinical profile)
    'true_label': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    'profile':  [
        'Hypertensive diabetic male, typical angina',
        'Post-menopausal female, non-anginal chest pain',
        'Young athletic male, atypical angina',
        'Elderly female, typical angina with ST depression',
        'Young male runner, asymptomatic',
        'Middle-aged female, atypical angina',
        'Older male, typical angina, ST depression',
        'Perimenopausal female, non-anginal',
        'Senior male, typical angina, 3-vessel disease',
        'Young female, non-anginal, low risk',
        'Elderly male, severe multi-vessel CAD',
        'Middle-aged female, mild atypical angina',
        'Young male, non-anginal, very low risk',
        'Senior female, typical angina, diabetic',
        'Middle-aged male, exertional chest pain',
        'Young female, atypical, low risk',
        'Elderly male, 3-vessel CAD, severe',
        'Middle-aged female, non-anginal, healthy',
        'Senior male, typical angina, 2-vessel',
        'Elderly female, atypical, borderline'
    ]
}

df_mock = pd.DataFrame(mock_data)
print("=" * 70)
print("  KARDIA — MOCK PATIENT INFERENCE TEST")
print("=" * 70)
print(f"  Testing {len(df_mock)} realistic synthetic patients\n")


# ── Feature Engineering (must match preprocess.py) ───────────────────────────
def engineer_features(df):
    df = df.copy()
    df['hr_reserve']      = df['thalach'] - (220 - df['age'])
    df['chol_age_ratio']  = df['chol'] / df['age']
    df['cp_sex']          = df['cp'] * df['sex']
    df['thalach_oldpeak'] = df['thalach'] * df['oldpeak']
    df['age_bucket']      = pd.cut(
        df['age'], bins=[0,40,55,65,120], labels=[0,1,2,3]
    ).astype(int)
    return df

raw_cols = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']

feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal',
                'hr_reserve','chol_age_ratio','cp_sex',
                'thalach_oldpeak','age_bucket']

# ── Load scaler + imputer ─────────────────────────────────────────────────────
with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
# Step 1: mock data has no missing values — skip imputer entirely
df_raw = df_mock[raw_cols].copy()

# Step 2: engineer features
df_engineered = engineer_features(df_raw)
X_features    = df_engineered[feature_cols].values.astype(np.float32)

# Step 3: scale (scaler was fitted on 18 engineered features)
X_scaled = scaler.transform(X_features).astype(np.float32)

# ── Load ECG encoder + extract embeddings ────────────────────────────────────
ecg_signals = np.load('data/ptbxl/processed/ecg_signals.npy')
ecg_labels  = np.load('data/ptbxl/processed/ecg_labels.npy')
if ecg_signals.shape[1] != 12:
    ecg_signals = ecg_signals.transpose(0, 2, 1)

ecg_model = ResNet1D(embedding_dim=256)
ecg_model.load_state_dict(
    torch.load('models/best_resnet1d.pt', weights_only=True))
ecg_model.eval()
for p in ecg_model.parameters():
    p.requires_grad = False

with torch.no_grad():
    ecg_embeddings = ecg_model(
        torch.tensor(ecg_signals), return_embedding=True).numpy()

# Label-matched ECG embedding per mock patient
rng = np.random.default_rng(99)
true_labels = df_mock['true_label'].values

ecg_embs = []
for label in true_labels:
    idx = np.where(ecg_labels == int(label))[0]
    ecg_embs.append(ecg_embeddings[rng.choice(idx)])
ecg_embs = np.array(ecg_embs, dtype=np.float32)

X_fused = np.concatenate([X_scaled, ecg_embs], axis=1).astype(np.float32)


# ── Load fusion model ─────────────────────────────────────────────────────────
class FusionHead(nn.Module):
    def __init__(self, input_dim=274, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.binary_head   = nn.Linear(64, 1)
        self.risk_head     = nn.Linear(64, 1)
        self.severity_head = nn.Linear(64, 4)

    def forward(self, x):
        feat = self.net(x)
        return (torch.sigmoid(self.binary_head(feat)),
                torch.sigmoid(self.risk_head(feat)) * 100,
                self.severity_head(feat))

fused_dim = X_fused.shape[1]
model     = FusionHead(input_dim=fused_dim)
model.load_state_dict(
    torch.load('models/best_fusion_v2.pt', weights_only=True))
model.eval()


# ── Run inference ─────────────────────────────────────────────────────────────
severity_map = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
risk_map     = lambda r: ('🟢 LOW' if r < 30 else '🟡 MEDIUM' if r < 60 else '🔴 HIGH')

results = []
with torch.no_grad():
    binary, risk, severity = model(torch.tensor(X_fused))
    probs   = binary.squeeze().numpy()
    risks   = risk.squeeze().numpy()
    sevs    = severity.argmax(dim=1).numpy()

for i in range(len(df_mock)):
    prob      = float(probs[i])
    risk_val  = float(risks[i])
    sev_label = severity_map[int(sevs[i])]
    pred      = 1 if prob >= 0.5 else 0
    true      = int(true_labels[i])
    correct   = pred == true

    results.append({
        'name'    : df_mock['name'][i],
        'profile' : df_mock['profile'][i],
        'prob'    : prob,
        'risk'    : risk_val,
        'severity': sev_label,
        'pred'    : pred,
        'true'    : true,
        'correct' : correct,
        'risk_cat': risk_map(risk_val)
    })


# ── Print results ─────────────────────────────────────────────────────────────
print(f"{'#':<3} {'Patient':<20} {'Risk Score':>10} {'Level':<10} {'Severity':<10} {'Pred':>6} {'True':>6} {'✓':>4}")
print("-" * 78)

correct_count = 0
for i, r in enumerate(results):
    tick = '✅' if r['correct'] else '❌'
    pred_label = 'DISEASE' if r['pred'] else 'HEALTHY'
    true_label_str = 'DISEASE' if r['true'] else 'HEALTHY'
    if r['correct']:
        correct_count += 1
    print(f"{i+1:<3} {r['name']:<20} {r['risk']:>8.1f}%  {r['risk_cat']:<14} {r['severity']:<10} {pred_label:>8} {true_label_str:>8}  {tick}")

accuracy = correct_count / len(results) * 100

print("-" * 78)
print(f"\n📊 MOCK PATIENT TEST SUMMARY")
print(f"   Total patients   : {len(results)}")
print(f"   Correct          : {correct_count}")
print(f"   Incorrect        : {len(results) - correct_count}")
print(f"   Accuracy         : {accuracy:.1f}%")

print(f"\n📋 DETAILED PATIENT REPORTS")
print("-" * 78)
for i, r in enumerate(results):
    status = '🔴 DISEASE DETECTED' if r['pred'] == 1 else '🟢 HEALTHY'
    match  = '✅ Correct' if r['correct'] else '❌ Incorrect'
    print(f"\nPatient {i+1}: {r['name']}")
    print(f"  Profile     : {r['profile']}")
    print(f"  Risk Score  : {r['risk']:.1f}/100  {r['risk_cat']}")
    print(f"  Severity    : {r['severity']}")
    print(f"  Prediction  : {status}  ({r['prob']*100:.1f}% confidence)")
    print(f"  Outcome     : {match}")

print(f"\n{'='*70}")
print(f"  FINAL ACCURACY ON MOCK PATIENTS: {accuracy:.1f}%")
print(f"{'='*70}")

# ── Save results to CSV ───────────────────────────────────────────────────────
df_results = pd.DataFrame([{
    'patient'        : r['name'],
    'profile'        : r['profile'],
    'risk_score'     : round(r['risk'], 1),
    'risk_category'  : r['risk_cat'].split()[-1],
    'severity'       : r['severity'],
    'predicted'      : 'DISEASE' if r['pred'] else 'HEALTHY',
    'true_label'     : 'DISEASE' if r['true'] else 'HEALTHY',
    'correct'        : r['correct'],
    'confidence_pct' : round(r['prob']*100, 1)
} for r in results])

os.makedirs('reports', exist_ok=True)
df_results.to_csv('reports/mock_patient_results.csv', index=False)
print(f"\n✅ Results saved to reports/mock_patient_results.csv")
# ── Threshold sensitivity analysis ───────────────────────────────────────────
print("\n📈 THRESHOLD SENSITIVITY (tabular-driven decision)")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Disease Recall':>15} {'Healthy Spec':>13}")
print("-" * 52)
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50]:
    preds_t    = [1 if float(probs[i]) >= thresh else 0 for i in range(len(results))]
    acc_t      = sum(p == t for p, t in zip(preds_t, true_labels)) / len(true_labels) * 100
    recall_t   = sum(p == 1 and t == 1 for p, t in zip(preds_t, true_labels)) / sum(true_labels) * 100
    spec_t     = sum(p == 0 and t == 0 for p, t in zip(preds_t, true_labels)) / (len(true_labels) - sum(true_labels)) * 100
    print(f"{thresh:>10.2f} {acc_t:>9.1f}% {recall_t:>14.1f}% {spec_t:>12.1f}%")