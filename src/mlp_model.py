import torch
import torch.nn as nn

class HeartMLP(nn.Module):
    def __init__(self, input_dim=18, dropout=0.3):
        super(HeartMLP, self).__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3 — 64-dim tabular embedding
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Output heads
        self.binary_head   = nn.Linear(64, 1)   # binary: disease yes/no
        self.risk_head     = nn.Linear(64, 1)   # risk score 0-100
        self.severity_head = nn.Linear(64, 4)   # severity: none/mild/mod/severe

    def forward(self, x):
        embedding = self.encoder(x)
        binary    = torch.sigmoid(self.binary_head(embedding))
        risk      = torch.sigmoid(self.risk_head(embedding)) * 100
        severity  = self.severity_head(embedding)
        return binary, risk, severity, embedding


if __name__ == '__main__':
    model = HeartMLP(input_dim=18)
    x     = torch.randn(8, 18)  # batch of 8
    b, r, s, e = model(x)
    print(f"Binary output   : {b.shape}")
    print(f"Risk score      : {r.shape}")
    print(f"Severity output : {s.shape}")
    print(f"Embedding       : {e.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")