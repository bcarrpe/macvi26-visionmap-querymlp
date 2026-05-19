"""
train_query_mlp.py v3

Inputs: [dist_norm, inv_dist, bearing_norm, pitch_norm, roll_norm, heading_norm]
Target: [cx, bottom_cy] in normalized image coordinates [0, 1]

Improvements over v2:
- Dropout + BatchNorm + 3 layers (in QueryMLP)
- Weight decay 1e-4 in AdamW
- Cosine annealing scheduler instead of StepLR

Run from /workspace:
    python train_query_mlp.py
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from query_mlp import QueryMLP

# ── config ────────────────────────────────────────────────────────────────────
YAML_FILE     = "dataset.yaml"
EPOCHS        = 1000
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
HIDDEN_DIM    = 128
INPUT_DIM     = 6
SEED          = 42
PATIENCE      = 60
PITCH_SCALE   = 10.0
ROLL_SCALE    = 10.0
HEADING_SCALE = 180.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── load dataset paths ────────────────────────────────────────────────────────
with open(YAML_FILE, 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

train_path = Path(data['train'])
val_path   = Path(data['val'])


def build_correspondences(data_path: Path):
    labels_dir  = data_path / "labels"
    queries_dir = data_path / "queries"
    imu_dir     = data_path / "imu"

    label_files = sorted(labels_dir.iterdir())
    query_files = sorted(queries_dir.iterdir())
    imu_files   = sorted(imu_dir.iterdir())

    inputs  = []
    targets = []
    skipped = 0

    for lf, qf, imuf in zip(label_files, query_files, imu_files):
        try:
            labels = np.loadtxt(lf)
        except Exception:
            skipped += 1
            continue
        if labels.size == 0:
            continue
        labels = labels.reshape(-1, 5)

        try:
            queries = np.loadtxt(qf).reshape(-1, 5)
        except Exception:
            skipped += 1
            continue

        try:
            imu = np.loadtxt(imuf).flatten()
        except Exception:
            skipped += 1
            continue

        pitch_deg   = float(imu[0])
        roll_deg    = float(imu[1])
        heading_deg = float(imu[2])

        label_dict = {int(row[0]): row[1:5] for row in labels}

        for q_row in queries:
            buoy_id = int(q_row[0])
            dist_m  = float(q_row[1])
            bearing = float(q_row[2])

            if buoy_id not in label_dict:
                continue

            cx, cy, w, h = label_dict[buoy_id]
            bottom_cy = float(np.clip(cy + h / 2.0, 0.0, 1.0))

            dist_norm = dist_m / 1000.0
            inv_dist  = float(np.clip(1.0 / max(dist_norm, 0.001), 0.0, 10.0))

            inputs.append([
                dist_norm,
                inv_dist,
                bearing     / 180.0,
                pitch_deg   / PITCH_SCALE,
                roll_deg    / ROLL_SCALE,
                heading_deg / HEADING_SCALE,
            ])
            targets.append([cx, bottom_cy])

    print(f"  {len(inputs)} correspondences ({skipped} skipped)")
    return (torch.tensor(inputs,  dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32))


# ── build datasets ────────────────────────────────────────────────────────────
print("Building training correspondences...")
X_train, Y_train = build_correspondences(train_path)
print("Building validation correspondences...")
X_val, Y_val = build_correspondences(val_path)
print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}")

# ── dataloaders ───────────────────────────────────────────────────────────────
train_loader = DataLoader(TensorDataset(X_train, Y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, Y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── model ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

mlp       = QueryMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.SmoothL1Loss()

# ── training ──────────────────────────────────────────────────────────────────
print(f"Training up to {EPOCHS} epochs (patience={PATIENCE})...")
best_val_loss     = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    mlp.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(mlp(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    mlp.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_losses.append(criterion(mlp(xb.to(device)), yb.to(device)).item())

    scheduler.step()
    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | train: {train_loss:.6f} | val: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(mlp.state_dict(), 'query_mlp.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print(f"\nBest val loss: {best_val_loss:.6f}")

# ── pixel error ───────────────────────────────────────────────────────────────
mlp.load_state_dict(torch.load('query_mlp.pth', map_location='cpu'))
mlp.eval()
with torch.no_grad():
    pred   = mlp(X_val.to(device)).cpu()
    err_px = torch.sqrt(((pred[:,0] - Y_val[:,0]) * 960)**2 +
                        ((pred[:,1] - Y_val[:,1]) * 540)**2)

print(f"\nPixel error on val set:")
print(f"  Mean:        {err_px.mean():.1f}px")
print(f"  Median:      {err_px.median():.1f}px")
print(f"  90th pct:    {err_px.quantile(0.9):.1f}px")
print(f"  Max:         {err_px.max():.1f}px")

# ── sample predictions ────────────────────────────────────────────────────────
print("\nSample predictions (first 10 val):")
print(f"{'dist':>7} {'1/d':>7} {'bear':>7} {'pitch':>7} {'roll':>7} {'head':>7} | "
      f"{'p_cx':>7} {'p_cy':>7} | {'t_cx':>7} {'t_cy':>7} | {'err':>7}")
with torch.no_grad():
    preds = mlp(X_val[:10].to(device)).cpu()
for i in range(10):
    feats = X_val[i].tolist()
    pcx, pcy = preds[i].tolist()
    tcx, tcy = Y_val[i].tolist()
    err = float(torch.sqrt(((preds[i,0]-Y_val[i,0])*960)**2 +
                            ((preds[i,1]-Y_val[i,1])*540)**2))
    print(f"{feats[0]:>7.3f} {feats[1]:>7.3f} {feats[2]:>7.3f} {feats[3]:>7.3f} "
          f"{feats[4]:>7.3f} {feats[5]:>7.3f} | "
          f"{pcx:>7.3f} {pcy:>7.3f} | {tcx:>7.3f} {tcy:>7.3f} | {err:>6.1f}px")

with open('query_mlp_val.txt', 'w') as f:
    f.write(f"Best val loss (SmoothL1): {best_val_loss:.6f}\n")
    f.write(f"Mean pixel error:   {err_px.mean():.1f}px\n")
    f.write(f"Median pixel error: {err_px.median():.1f}px\n")
    f.write(f"90th percentile:    {err_px.quantile(0.9):.1f}px\n")
    f.write(f"Train samples: {len(X_train)}\n")
    f.write(f"Val samples:   {len(X_val)}\n")