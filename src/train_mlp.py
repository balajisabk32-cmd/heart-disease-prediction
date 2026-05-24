import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics  import roc_auc_score, f1_score, accuracy_score
from mlp_model        import HeartMLP

# ── 1. Load data ──────────────────────────────────────────────────────────────
X_train = pd.read_csv('data/tabular/processed/X_train.csv').values.astype(np.float32)
X_val   = pd.read_csv('data/tabular/processed/X_val.csv').values.astype(np.float32)
X_test  = pd.read_csv('data/tabular/processed/X_test.csv').values.astype(np.float32)
y_train = pd.read_csv('data/tabular/processed/y_train.csv').values.ravel().astype(np.float32)
y_val   = pd.read_csv('data/tabular/processed/y_val.csv').values.ravel().astype(np.float32)
y_test  = pd.read_csv('data/tabular/processed/y_test.csv').values.ravel().astype(np.float32)

# Severity labels: map binary → 4-class (0=none,1=mild,2=mod,3=severe)
# For now we use binary as proxy — Phase 6 adds real severity from UCI targets
y_train_sev = (y_train * 1).astype(np.int64)  # placeholder
y_val_sev   = (y_val   * 1).astype(np.int64)
y_test_sev  = (y_test  * 1).astype(np.int64)

# ── 2. DataLoaders ────────────────────────────────────────────────────────────
def make_loader(X, y_bin, y_sev, shuffle=False, batch=32):
    ds = TensorDataset(
        torch.tensor(X),
        torch.tensor(y_bin).unsqueeze(1),
        torch.tensor(y_sev)
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, y_train_sev, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   y_val_sev)
test_loader  = make_loader(X_test,  y_test,  y_test_sev)

# ── 3. Model, loss, optimizer ─────────────────────────────────────────────────
device = torch.device('cpu')
model  = HeartMLP(input_dim=18, dropout=0.3).to(device)

binary_loss   = nn.BCELoss()
severity_loss = nn.CrossEntropyLoss()
optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ── 4. Training loop ──────────────────────────────────────────────────────────
EPOCHS       = 150
best_val_auc = 0.0
patience     = 20
no_improve   = 0
history      = {'train_loss':[], 'val_loss':[], 'val_auc':[]}

print("Starting MLP training...")
print(f"Device: {device} | Epochs: {EPOCHS} | Batch: 32\n")

for epoch in range(1, EPOCHS + 1):
    # — Train —
    model.train()
    train_loss = 0.0
    for X_b, y_bin_b, y_sev_b in train_loader:
        X_b, y_bin_b, y_sev_b = X_b.to(device), y_bin_b.to(device), y_sev_b.to(device)
        optimizer.zero_grad()
        b_out, r_out, s_out, _ = model(X_b)
        loss = binary_loss(b_out, y_bin_b) + 0.3 * severity_loss(s_out, y_sev_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # — Validate —
    model.eval()
    val_loss  = 0.0
    all_probs = []
    all_true  = []
    with torch.no_grad():
        for X_b, y_bin_b, y_sev_b in val_loader:
            b_out, r_out, s_out, _ = model(X_b)
            loss = binary_loss(b_out, y_bin_b) + 0.3 * severity_loss(s_out, y_sev_b)
            val_loss += loss.item()
            all_probs.extend(b_out.squeeze().numpy())
            all_true.extend(y_bin_b.squeeze().numpy())

    val_auc = roc_auc_score(all_true, all_probs)
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss   / len(val_loader))
    history['val_auc'].append(val_auc)
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss/len(val_loader):.4f} | Val AUC: {val_auc:.4f}")

    # — Early stopping & checkpoint —
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'models/best_mlp.pt')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} — best Val AUC: {best_val_auc:.4f}")
            break

# ── 5. Test evaluation ────────────────────────────────────────────────────────
model.load_state_dict(torch.load('models/best_mlp.pt', weights_only=True))
model.eval()

all_probs, all_preds, all_true = [], [], []
with torch.no_grad():
    for X_b, y_bin_b, y_sev_b in test_loader:
        b_out, _, _, _ = model(X_b)
        probs = b_out.squeeze().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(y_bin_b.squeeze().numpy())

test_auc = roc_auc_score(all_true, all_probs)
test_f1  = f1_score(all_true, all_preds)
test_acc = accuracy_score(all_true, all_preds)

print(f"\n{'='*50}")
print(f"  MLP TEST RESULTS")
print(f"{'='*50}")
print(f"  AUC      : {test_auc:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1 Score : {test_f1:.4f}")
print(f"\n  Baseline to beat (LR): AUC=0.8514 | F1=0.7778")
print(f"  Improvement in AUC   : {test_auc - 0.8514:+.4f}")

# ── 6. Training curves ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train Loss', color='steelblue')
ax1.plot(history['val_loss'],   label='Val Loss',   color='coral')
ax1.set_title('Training & Validation Loss')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history['val_auc'], color='green', label='Val AUC')
ax2.axhline(y=best_val_auc, color='red', linestyle='--',
            label=f'Best AUC: {best_val_auc:.4f}')
ax2.set_title('Validation AUC over Epochs')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('reports/mlp_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Training curves saved")
print("✅ Best model saved to models/best_mlp.pt")