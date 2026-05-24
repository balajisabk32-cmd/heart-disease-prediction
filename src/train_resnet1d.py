import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data    import DataLoader, TensorDataset, random_split
from sklearn.metrics     import roc_auc_score, f1_score, accuracy_score
from resnet1d            import ResNet1D

# ── 1. Load preprocessed ECG data ────────────────────────────────────────────
signals = np.load('data/ptbxl/processed/ecg_signals.npy')  # (200, 1000, 12)
labels  = np.load('data/ptbxl/processed/ecg_labels.npy')   # (200,)

# Transpose: (N, 1000, 12) → (N, 12, 1000) for Conv1D (channels first)
signals = signals.transpose(0, 2, 1).astype(np.float32)
labels  = labels.astype(np.float32)

print(f"Signals shape : {signals.shape}")
print(f"Labels shape  : {labels.shape}")
print(f"Class balance : {labels.mean()*100:.1f}% disease")

# ── 2. Train / Val / Test split ───────────────────────────────────────────────
X = torch.tensor(signals)
y = torch.tensor(labels).unsqueeze(1)

dataset    = TensorDataset(X, y)
n_total    = len(dataset)
n_train    = int(0.70 * n_total)
n_val      = int(0.15 * n_total)
n_test     = n_total - n_train - n_val

torch.manual_seed(42)
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16)
test_loader  = DataLoader(test_ds,  batch_size=16)

print(f"\nSplit — Train: {n_train} | Val: {n_val} | Test: {n_test}")

# ── 3. Model, loss, optimizer ─────────────────────────────────────────────────
device    = torch.device('cpu')
model     = ResNet1D(embedding_dim=256).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters : {total_params:,}")

# ── 4. Training loop ──────────────────────────────────────────────────────────
EPOCHS       = 100
best_val_auc = 0.0
patience     = 15
no_improve   = 0
history      = {'train_loss': [], 'val_loss': [], 'val_auc': []}

print(f"\nTraining ResNet1D on CPU...")
print(f"Epochs: {EPOCHS} | Batch: 16 | LR: 5e-4\n")

for epoch in range(1, EPOCHS + 1):

    # — Train —
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out, _ = model(X_b)
        loss   = criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    # — Validate —
    model.eval()
    val_loss  = 0.0
    all_probs = []
    all_true  = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out, _ = model(X_b)
            loss   = criterion(out, y_b)
            val_loss  += loss.item()
            all_probs.extend(out.squeeze().numpy())
            all_true.extend(y_b.squeeze().numpy())

    val_auc = roc_auc_score(all_true, all_probs)
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss     / len(val_loader))
    history['val_auc'].append(val_auc)
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss/len(val_loader):.4f} | Val AUC: {val_auc:.4f}")

    # — Checkpoint + early stopping —
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'models/best_resnet1d.pt')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} — best Val AUC: {best_val_auc:.4f}")
            break

# ── 5. Test evaluation ────────────────────────────────────────────────────────
model.load_state_dict(torch.load('models/best_resnet1d.pt', weights_only=True))
model.eval()

all_probs, all_preds, all_true = [], [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        out, _ = model(X_b)
        probs  = out.squeeze().numpy()
        preds  = (probs >= 0.5).astype(int)
        all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
        all_preds.extend(preds if preds.ndim > 0 else [preds.item()])
        all_true.extend(y_b.squeeze().numpy())

test_auc = roc_auc_score(all_true, all_probs)
test_f1  = f1_score(all_true, all_preds, zero_division=0)
test_acc = accuracy_score(all_true, all_preds)

print(f"\n{'='*50}")
print(f"  RESNET1D ECG-ONLY TEST RESULTS")
print(f"{'='*50}")
print(f"  AUC      : {test_auc:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1 Score : {test_f1:.4f}")
print(f"  Best Val AUC : {best_val_auc:.4f}")

# ── 6. Training curves ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train Loss', color='steelblue')
ax1.plot(history['val_loss'],   label='Val Loss',   color='coral')
ax1.set_title('ResNet1D — Loss Curves')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history['val_auc'], color='green', label='Val AUC')
ax2.axhline(y=best_val_auc, color='red', linestyle='--',
            label=f'Best: {best_val_auc:.4f}')
ax2.set_title('ResNet1D — Validation AUC')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('reports/resnet1d_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Training curves saved to reports/")
print("✅ Best ECG encoder saved to models/best_resnet1d.pt")
print("\n→ Next: Multimodal Fusion (Phase 7)")