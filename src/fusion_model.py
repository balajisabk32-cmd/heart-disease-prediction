import torch
import torch.nn as nn
from mlp_model  import HeartMLP
from resnet1d   import ResNet1D


class MultimodalFusion(nn.Module):
    """
    Attention-gated multimodal fusion.

    Inputs:
        x_tab : (batch, 18)        — tabular clinical features
        x_ecg : (batch, 12, 1000)  — 12-lead ECG signal

    Outputs:
        binary   : (batch, 1)   — disease probability 0-1
        risk     : (batch, 1)   — risk score 0-100
        severity : (batch, 4)   — severity logits
        alpha    : scalar       — attention weight on tabular
        beta     : scalar       — attention weight on ECG
    """
    def __init__(self, tab_input_dim=18, ecg_embedding_dim=256,
                 tab_embedding_dim=64, fusion_dim=128, dropout=0.3):
        super().__init__()

        # ── Tabular encoder (from HeartMLP, encoder only) ─────────────────
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, tab_embedding_dim),
            nn.BatchNorm1d(tab_embedding_dim),
            nn.ReLU(),
        )

        # ── ECG encoder (ResNet1D, embedding only) ────────────────────────
        self.ecg_encoder = ResNet1D(embedding_dim=ecg_embedding_dim)

        # Project ECG 256-dim → 64-dim to match tabular embedding
        self.ecg_projector = nn.Sequential(
            nn.Linear(ecg_embedding_dim, tab_embedding_dim),
            nn.BatchNorm1d(tab_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Attention gate — learnable modality weights ───────────────────
        # Takes both embeddings, outputs 2 weights (alpha for tab, beta for ECG)
        self.attention_gate = nn.Sequential(
            nn.Linear(tab_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),   # alpha + beta = 1
        )

        # ── Fusion trunk ──────────────────────────────────────────────────
        self.fusion_trunk = nn.Sequential(
            nn.Linear(tab_embedding_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # ── Three output heads ────────────────────────────────────────────
        self.binary_head   = nn.Linear(64, 1)
        self.risk_head     = nn.Linear(64, 1)
        self.severity_head = nn.Linear(64, 4)

    def forward(self, x_tab, x_ecg):
        # Encode each modality
        tab_emb = self.tab_encoder(x_tab)                        # (B, 64)
        ecg_emb = self.ecg_encoder(x_ecg, return_embedding=True) # (B, 256)
        ecg_emb = self.ecg_projector(ecg_emb)                    # (B, 64)

        # Attention gate — how much to trust each modality per sample
        gate_input  = torch.cat([tab_emb, ecg_emb], dim=1)       # (B, 128)
        attn_weights = self.attention_gate(gate_input)            # (B, 2)
        alpha = attn_weights[:, 0:1]  # tabular weight
        beta  = attn_weights[:, 1:2]  # ECG weight

        # Weighted combination
        fused = torch.cat([alpha * tab_emb, beta * ecg_emb], dim=1)  # (B, 128)

        # Shared trunk
        trunk_out = self.fusion_trunk(fused)  # (B, 64)

        # Output heads
        binary   = torch.sigmoid(self.binary_head(trunk_out))
        risk     = torch.sigmoid(self.risk_head(trunk_out)) * 100
        severity = self.severity_head(trunk_out)

        return binary, risk, severity, alpha.mean(), beta.mean()


# ── Quick architecture test ───────────────────────────────────────────────────
if __name__ == '__main__':
    model   = MultimodalFusion(tab_input_dim=18)
    x_tab   = torch.randn(8, 18)
    x_ecg   = torch.randn(8, 12, 1000)

    binary, risk, severity, alpha, beta = model(x_tab, x_ecg)

    print(f"Binary output    : {binary.shape}")
    print(f"Risk score       : {risk.shape}")
    print(f"Severity output  : {severity.shape}")
    print(f"Alpha (tabular)  : {alpha.item():.4f}")
    print(f"Beta  (ECG)      : {beta.item():.4f}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")