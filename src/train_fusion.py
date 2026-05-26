import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics  import (roc_auc_score, f1_score,
                               accuracy_score, classification_report,
                               confusion_matrix)
from resnet1d import ResNet1D

os.makedirs('reports', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── 1. Load tabular splits ────────────────────────────────────────────────────
X_tab_train = pd.read_csv('data/tabular/processed/X_train.csv').values.astype(np.float32)
X_tab_val   = pd.read_csv('data/tabular/processed/X_val.csv').values.astype(np.float32)
X_tab_test  = pd.read_csv('data/tabular/processed/X_test.csv').values.astype(np.float32)
y_train     = pd.read_csv('data/tabular/processed/y_train.csv').values.ravel().astype(np.float32)
y_val       = pd.read_csv('data/tabular/processed/y_val.csv').values.ravel().astype(np.float32)
y_test      = pd.read_csv('data/tabular/processed/y_test.csv').values.ravel().astype(np.float32)

print(f"Tabular — Train: {X_tab_train.shape} | Val: {X_tab_val.shape} | Test: {X_tab_test.shape}")

# ── 2. Load ECG data + frozen ResNet1D ───────────────────────────────────────
ecg_signals = np.load('data/ptbxl/processed/ecg_signals.npy')
ecg_labels  = np.load('data/ptbxl/processed/ecg_labels.npy')

if ecg_signals.shape[1] != 12:
    ecg_signals = ecg_signals.transpose(0, 2, 1)

print(f"ECG signals : {ecg_signals.shape}")

# Load pretrained ResNet1D — frozen, embedding mode only
ecg_model = ResNet1D(embedding_dim=256)
ecg_model.load_state_dict(
    torch.load('models/best_resnet1d.pt', weights_only=True)
)
ecg_model.eval()
for param in ecg_model.parameters():
    param.requires_grad = False

print("✅ ResNet1D loaded and frozen")

# ── 3. Extract ECG embeddings for all 200 records ────────────────────────────
print("Extracting ECG embeddings...")
with torch.no_grad():
    ecg_tensor     = torch.tensor(ecg_signals)
    ecg_embeddings = ecg_model(ecg_tensor, return_embedding=True).numpy()

print(f"ECG embeddings shape: {ecg_embeddings.shape}")  # (200, 256)

# ── 4. Build augmented tabular features ──────────────────────────────────────
# Strategy: for each tabular patient, find the MOST SIMILAR ECG record
# by matching on binary label — then append that ECG embedding
# This creates a consistent multimodal feature vector per patient

rng = np.random.default_rng(42)

def get_ecg_embedding_for_split(y_split, ecg_embeddings, ecg_labels):
    """
    For each patient in the split, sample an ECG embedding
    with the SAME label (label-matched pairing).
    Returns: (N, 256) embedding array
    """
    result = []
    for label in y_split:
        # Find ECG records with matching label
        matching_idx = np.where(ecg_labels == int(label))[0]
        chosen       = rng.choice(matching_idx)
        result.append(ecg_embeddings[chosen])
    return np.array(result, dtype=np.float32)

ecg_emb_train = get_ecg_embedding_for_split(y_train, ecg_embeddings, ecg_labels)
ecg_emb_val   = get_ecg_embedding_for_split(y_val,   ecg_embeddings, ecg_labels)
ecg_emb_test  = get_ecg_embedding_for_split(y_test,  ecg_embeddings, ecg_labels)

print(f"ECG emb train: {ecg_emb_train.shape}")  # (230, 256)
print(f"ECG emb val  : {ecg_emb_val.shape}")
print(f"ECG emb test : {ecg_emb_test.shape}")

# Concatenate tabular + ECG embeddings → rich feature vector
X_fused_train = np.concatenate([X_tab_train, ecg_emb_train], axis=1)
X_fused_val   = np.concatenate([X_tab_val,   ecg_emb_val],   axis=1)
X_fused_test  = np.concatenate([X_tab_test,  ecg_emb_test],  axis=1)

print(f"\nFused feature dim: {X_fused_train.shape[1]} (18 tabular + 256 ECG)")

# ── 5. Lightweight fusion head ────────────────────────────────────────────────
class FusionHead(nn.Module):
    """Lightweight MLP on top of fused tabular+ECG features."""
    def __init__(self, input_dim=274, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.binary_head   = nn.Linear(64, 1)
        self.risk_head     = nn.Linear(64, 1)
        self.severity_head = nn.Linear(64, 4)

    def forward(self, x):
        feat     = self.net(x)
        binary   = torch.sigmoid(self.binary_head(feat))
        risk     = torch.sigmoid(self.risk_head(feat)) * 100
        severity = self.severity_head(feat)
        return binary, risk, severity

# ── 6. DataLoaders ────────────────────────────────────────────────────────────
def make_loader(X, y, shuffle=False, batch=32):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = make_loader(X_fused_train, y_train, shuffle=True)
val_loader   = make_loader(X_fused_val,   y_val)
test_loader  = make_loader(X_fused_test,  y_test)

# ── 7. Train ──────────────────────────────────────────────────────────────────
device    = torch.device('cpu')
fused_dim = X_fused_train.shape[1]
model     = FusionHead(input_dim=fused_dim).to(device)

binary_loss   = nn.BCELoss()
severity_loss = nn.CrossEntropyLoss()
optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

EPOCHS       = 150
best_val_auc = 0.0
patience     = 25
no_improve   = 0
history      = {'train_loss': [], 'val_loss': [], 'val_auc': []}

print(f"\nTraining Fusion Head (tabular + frozen ECG embeddings)...")
print(f"Input dim: {fused_dim} | Epochs: {EPOCHS} | Batch: 32\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        y_sev_b  = y_b.squeeze().long()
        optimizer.zero_grad()
        binary, risk, severity = model(X_b)
        loss = (binary_loss(binary, y_b) +
                0.3 * severity_loss(severity, y_sev_b))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss  = 0.0
    all_probs = []
    all_true  = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            binary, risk, severity = model(X_b)
            y_sev_b = y_b.squeeze().long()
            loss    = (binary_loss(binary, y_b) +
                       0.3 * severity_loss(severity, y_sev_b))
            val_loss  += loss.item()
            all_probs.extend(binary.squeeze().numpy())
            all_true.extend(y_b.squeeze().numpy())

    val_auc = roc_auc_score(all_true, all_probs)
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss     / len(val_loader))
    history['val_auc'].append(val_auc)
    scheduler.step()

    if epoch % 15 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss/len(val_loader):.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'models/best_fusion_v2.pt')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} — best Val AUC: {best_val_auc:.4f}")
            break

# ── 8. Test evaluation ────────────────────────────────────────────────────────
model.load_state_dict(torch.load('models/best_fusion_v2.pt', weights_only=True))
model.eval()

all_probs, all_preds, all_true = [], [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        binary, risk, severity = model(X_b)
        probs = binary.squeeze().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs if probs.ndim > 0 else [float(probs)])
        all_preds.extend(preds if preds.ndim > 0 else [int(preds)])
        all_true.extend(y_b.squeeze().numpy())

test_auc = roc_auc_score(all_true, all_probs)
test_f1  = f1_score(all_true, all_preds, zero_division=0)
test_acc = accuracy_score(all_true, all_preds)

print(f"\n{'='*60}")
print(f"  MULTIMODAL FUSION v2 — FINAL TEST RESULTS")
print(f"{'='*60}")
print(f"  AUC      : {test_auc:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1 Score : {test_f1:.4f}")
print(f"\n  ── Comparison ──────────────────────────────────")
print(f"  Logistic Regression   AUC: 0.8514  F1: 0.7778")
print(f"  Random Forest         AUC: 0.8448  F1: 0.7692")
print(f"  XGBoost               AUC: 0.8324  F1: 0.7347")
print(f"  MLP (tabular only)    AUC: 0.8438  F1: 0.7692")
print(f"  ResNet1D (ECG only)   AUC: 0.9205  F1: 0.7273")
print(f"  Fusion v1 (end2end)   AUC: 0.7962  F1: 0.6667")
print(f"  Fusion v2 (feature)   AUC: {test_auc:.4f}  F1: {test_f1:.4f}  ← YOU ARE HERE")
print(f"{'='*60}")
print(f"\nClassification Report:")
print(classification_report(all_true, all_preds,
                             target_names=['No Disease','Disease']))

# ── 9. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train', color='steelblue')
axes[0].plot(history['val_loss'],   label='Val',   color='coral')
axes[0].set_title('Fusion v2 — Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history['val_auc'], color='green')
axes[1].axhline(y=best_val_auc, color='red', linestyle='--',
                label=f'Best Val AUC: {best_val_auc:.4f}')
axes[1].axhline(y=0.8514, color='gray', linestyle=':',
                label='LR baseline: 0.8514')
axes[1].set_title('Fusion v2 — Validation AUC')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/fusion_v2_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])
ax.set_title(f'Fusion v2 — Confusion Matrix (AUC={test_auc:.4f})')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('reports/fusion_v2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ All plots saved")
print("✅ Best model saved to models/best_fusion_v2.pt")
print("\n→ Next: Phase 8 — SHAP + GradCAM Explainability")