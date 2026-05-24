import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import os
import pickle
from resnet1d  import ResNet1D

os.makedirs('explainability', exist_ok=True)

# ── 1. Load everything ────────────────────────────────────────────────────────
feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal',
                 'hr_reserve','chol_age_ratio','cp_sex',
                 'thalach_oldpeak','age_bucket']

X_train = pd.read_csv('data/tabular/processed/X_train.csv').values.astype(np.float32)
X_test  = pd.read_csv('data/tabular/processed/X_test.csv').values.astype(np.float32)
y_test  = pd.read_csv('data/tabular/processed/y_test.csv').values.ravel()

ecg_signals = np.load('data/ptbxl/processed/ecg_signals.npy')
ecg_labels  = np.load('data/ptbxl/processed/ecg_labels.npy')
if ecg_signals.shape[1] != 12:
    ecg_signals = ecg_signals.transpose(0, 2, 1)

print("✅ Data loaded")

# ── 2. Load frozen ECG encoder + extract embeddings ──────────────────────────
ecg_model = ResNet1D(embedding_dim=256)
ecg_model.load_state_dict(
    torch.load('models/best_resnet1d.pt', weights_only=True))
ecg_model.eval()
for p in ecg_model.parameters():
    p.requires_grad = False

with torch.no_grad():
    ecg_embeddings = ecg_model(
        torch.tensor(ecg_signals), return_embedding=True).numpy()

rng = np.random.default_rng(42)

def get_matched_embeddings(y_split):
    result = []
    for label in y_split:
        idx = np.where(ecg_labels == int(label))[0]
        result.append(ecg_embeddings[rng.choice(idx)])
    return np.array(result, dtype=np.float32)

ecg_emb_train = get_matched_embeddings(
    pd.read_csv('data/tabular/processed/y_train.csv').values.ravel())
ecg_emb_test  = get_matched_embeddings(y_test)

X_fused_train = np.concatenate([X_train, ecg_emb_train], axis=1).astype(np.float32)
X_fused_test  = np.concatenate([X_test,  ecg_emb_test],  axis=1).astype(np.float32)

print(f"Fused train: {X_fused_train.shape} | Fused test: {X_fused_test.shape}")

# ── 3. Load fusion model ──────────────────────────────────────────────────────
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

fused_dim = X_fused_train.shape[1]
fusion    = FusionHead(input_dim=fused_dim)
fusion.load_state_dict(
    torch.load('models/best_fusion_v2.pt', weights_only=True))
fusion.eval()
print("✅ Fusion model loaded")

# ── 4. SHAP — tabular feature importance ─────────────────────────────────────
print("\nComputing SHAP values (this takes ~2 minutes)...")

# Wrapper: SHAP needs a function that takes numpy → numpy probability
def predict_proba(X_np):
    with torch.no_grad():
        out, _, _ = fusion(torch.tensor(X_np.astype(np.float32)))
    return out.squeeze().numpy()

# Use KernelExplainer on tabular features only (first 18 dims)
# Background = mean of training set (summarised to 50 samples for speed)
background   = shap.kmeans(X_fused_train[:, :18], 50)

# Wrapper for tabular-only SHAP (append mean ECG embedding)
mean_ecg_emb = X_fused_train[:, 18:].mean(axis=0, keepdims=True)

def predict_tabular_only(X_tab_np):
    ecg_part = np.tile(mean_ecg_emb, (X_tab_np.shape[0], 1))
    X_full   = np.concatenate([X_tab_np, ecg_part], axis=1).astype(np.float32)
    with torch.no_grad():
        out, _, _ = fusion(torch.tensor(X_full))
    result = out.squeeze().numpy()
    return np.atleast_1d(result)  # always 1D even for batch size 1

explainer   = shap.KernelExplainer(predict_tabular_only, background)
shap_values = explainer.shap_values(X_fused_test[:, :18], nsamples=100)

print("✅ SHAP values computed")

# ── 5. SHAP Summary Plot ──────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_fused_test[:, :18],
    feature_names=feature_names,
    show=False,
    plot_type='dot'
)
plt.title('SHAP Feature Importance — Multimodal Fusion Model', fontsize=13)
plt.tight_layout()
plt.savefig('explainability/shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ SHAP summary plot saved")

# ── 6. SHAP Bar Plot — global importance ─────────────────────────────────────
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_fused_test[:, :18],
    feature_names=feature_names,
    show=False,
    plot_type='bar'
)
plt.title('SHAP Global Feature Importance', fontsize=13)
plt.tight_layout()
plt.savefig('explainability/shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ SHAP bar plot saved")

# ── 7. SHAP Waterfall — single patient explanation ───────────────────────────
print("\nGenerating per-patient SHAP waterfall plots...")

for patient_idx in [0, 1, 2]:
    patient_tab  = X_fused_test[patient_idx:patient_idx+1, :18]
    ecg_part     = mean_ecg_emb
    patient_full = np.concatenate([patient_tab, ecg_part], axis=1).astype(np.float32)

    with torch.no_grad():
        prob, risk, sev = fusion(torch.tensor(patient_full))

    prob_val = prob.item()
    risk_val = risk.item()
    true_val = int(y_test[patient_idx])
    pred_val = 1 if prob_val >= 0.5 else 0
    severity_map = {0:'None', 1:'Mild', 2:'Moderate', 3:'Severe'}
    sev_label    = severity_map[sev.argmax().item()]

    sv = shap_values[patient_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Patient {patient_idx+1} — Prediction: {"DISEASE" if pred_val else "HEALTHY"} '
        f'(True: {"DISEASE" if true_val else "HEALTHY"})\n'
        f'Risk Score: {risk_val:.1f}/100 | Severity: {sev_label} | '
        f'Confidence: {prob_val*100:.1f}%',
        fontsize=12, fontweight='bold'
    )

    # SHAP waterfall (manual bar chart)
    ax = axes[0]
    sorted_idx  = np.argsort(np.abs(sv))[::-1][:10]
    top_features = [feature_names[i] for i in sorted_idx]
    top_values   = sv[sorted_idx]
    colors       = ['#F44336' if v > 0 else '#4CAF50' for v in top_values]
    ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title('Top 10 Feature Contributions\n(Red=increases risk, Green=decreases risk)')
    ax.set_xlabel('SHAP Value')

    # Patient feature values
    ax2 = axes[1]
    patient_vals = X_fused_test[patient_idx, :18]
    top_feat_vals = [(feature_names[i], patient_vals[i], sv[i])
                     for i in sorted_idx]
    rows  = [f[0] for f in top_feat_vals]
    vals  = [f'{f[1]:.3f}' for f in top_feat_vals]
    shaps = [f'{f[2]:+.4f}' for f in top_feat_vals]

    table_data = list(zip(vals, shaps))
    table = ax2.table(
        cellText=table_data,
        rowLabels=rows,
        colLabels=['Feature Value', 'SHAP Impact'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax2.axis('off')
    ax2.set_title('Feature Values & SHAP Impact')

    plt.tight_layout()
    fname = f'explainability/patient_{patient_idx+1}_explanation.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Patient {patient_idx+1} explanation saved → {fname}")

# ── 8. GradCAM on ECG signal ─────────────────────────────────────────────────
print("\nGenerating GradCAM on ECG signals...")

class GradCAM1D:
    """GradCAM for 1D CNN — highlights which ECG time segments drove prediction."""
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def compute(self, x):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out, emb = self.model(x)
        self.model.zero_grad()
        out.squeeze().backward()
        # Weight activations by gradients
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam     = (weights * self.activations).sum(dim=1)   # (1, T)
        cam     = torch.relu(cam).squeeze().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, out.item()

ecg_model_grad = ResNet1D(embedding_dim=256)
ecg_model_grad.load_state_dict(
    torch.load('models/best_resnet1d.pt', weights_only=True))
ecg_model_grad.train()  # needed for gradients

gradcam = GradCAM1D(ecg_model_grad, ecg_model_grad.stage3[-1].conv_block[-2])

lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

for sample_idx in [0, 1]:
    ecg_sample = torch.tensor(ecg_signals[sample_idx:sample_idx+1])
    cam, prob  = gradcam.compute(ecg_sample)
    true_label = int(ecg_labels[sample_idx])

    # Upsample CAM to signal length
    cam_upsampled = np.interp(
        np.linspace(0, len(cam)-1, ecg_signals.shape[2]),
        np.arange(len(cam)), cam
    )

    time_axis = np.linspace(0, 10, ecg_signals.shape[2])

    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    fig.suptitle(
        f'GradCAM ECG Activation Map — Sample {sample_idx+1}\n'
        f'Prediction: {"DISEASE" if prob>=0.5 else "HEALTHY"} '
        f'({prob*100:.1f}%) | True: {"DISEASE" if true_label else "HEALTHY"}',
        fontsize=13, fontweight='bold'
    )

    for lead_idx, ax in enumerate(axes.flat):
        signal = ecg_signals[sample_idx, lead_idx, :]
        ax.plot(time_axis, signal, color='steelblue', linewidth=0.8, label='ECG')
        ax.fill_between(time_axis, signal.min(), signal.max(),
                        alpha=cam_upsampled * 0.5,
                        color='red', label='GradCAM')
        ax.set_title(f'Lead {lead_names[lead_idx]}', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fname = f'explainability/gradcam_ecg_sample_{sample_idx+1}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ GradCAM plot saved → {fname}")

# ── 9. Final summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  EXPLAINABILITY COMPLETE")
print(f"{'='*60}")
print(f"  explainability/shap_summary.png          — global feature importance")
print(f"  explainability/shap_bar.png              — ranked feature importance")
print(f"  explainability/patient_1_explanation.png — patient 1 SHAP waterfall")
print(f"  explainability/patient_2_explanation.png — patient 2 SHAP waterfall")
print(f"  explainability/patient_3_explanation.png — patient 3 SHAP waterfall")
print(f"  explainability/gradcam_ecg_sample_1.png  — ECG GradCAM heatmap")
print(f"  explainability/gradcam_ecg_sample_2.png  — ECG GradCAM heatmap")
print(f"{'='*60}")
print(f"\n✅ Phase 8 complete — model is fully explainable")
print(f"→ Next: Phase 9 — README + GitHub final push")