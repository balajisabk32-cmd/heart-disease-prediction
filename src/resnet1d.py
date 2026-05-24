import torch
import torch.nn as nn


# ── Building block ────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """One residual block: two conv layers with a skip connection."""
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        pad = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # Skip connection — match dimensions if needed
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))


# ── Full ResNet1D ─────────────────────────────────────────────────────────────
class ResNet1D(nn.Module):
    """
    ResNet1D ECG encoder.
    Input : (batch, 12, 1000)  — 12 leads × 1000 timepoints
    Output: 256-dim embedding vector per patient
    """
    def __init__(self, embedding_dim=256):
        super().__init__()

        # Stem — initial conv to extract low-level features
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages — progressively deeper features
        self.stage1 = nn.Sequential(
            ResBlock1D(64,  64),
            ResBlock1D(64,  64),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(64,  128, stride=2),
            ResBlock1D(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResBlock1D(128, 256, stride=2),
            ResBlock1D(256, 256),
        )

        # Global average pooling → fixed-size embedding regardless of input length
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classification head (used during ECG-only pre-training)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x, return_embedding=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        emb = self.embedding(x)

        if return_embedding:
            return emb                              # 256-dim vector for fusion

        out = torch.sigmoid(self.classifier(emb))  # binary ECG prediction
        return out, emb


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model   = ResNet1D(embedding_dim=256)
    x       = torch.randn(8, 12, 1000)   # batch=8, 12 leads, 1000 timepoints
    out, emb = model(x)

    print(f"Input shape      : {x.shape}")
    print(f"Output shape     : {out.shape}")
    print(f"Embedding shape  : {emb.shape}")
    print(f"Total parameters : {sum(p.numel() for p in model.parameters()):,}")