#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kiểm tra overfit của checkpoint chính thức: so sánh Train / Val / Test.

Chạy từ thư mục gốc repo (đã kích hoạt môi trường):
    python workspaces/member3_sensor/02_train_unimodal_sensor/check_overfit.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_unimodal_sensor as T  # noqa: E402

root = T.find_repo_root()
P = T.import_preprocess(root, None)
results_dir = HERE / "results"
ck_path = root / "checkpoints" / "unimodal" / "best_sensor_unimodal.pth"
ck = torch.load(ck_path, map_location="cpu", weights_only=False)

pre_path = Path(ck["preprocessor_path"])
if not pre_path.exists():
    pre_path = results_dir / "sensor_preprocessor_unimodal.joblib"
pre = P.SensorPreprocessor.load(pre_path)

net = P.SensorBackbone(input_dim=ck["input_dim"], hidden_dims=ck["hidden_dims"], latent_dim=ck["latent_dim"],
                       num_classes=ck["num_classes"], dropout=ck["dropout"])
net.load_state_dict(ck["full_state_dict"])
net.eval()

rows = []
for split in ("train", "valid", "test"):
    df = T.load_split(P, root / "data" / "final", split, results_dir, rebuild=False)
    X, y = pre.to_tensor(df), pre.encode_labels(df)
    with torch.no_grad():
        logits = net(X)
    pred = logits.argmax(1).numpy()
    rows.append((split, len(df), accuracy_score(y.numpy(), pred), f1_score(y.numpy(), pred, average="macro"),
                 F.cross_entropy(logits, y).item()))

print(f"\nCheckpoint: {ck_path.name} | bộ đặc trưng: {ck['feature_set']} | seed {ck['seed']}")
print(f"{'tập':<7}{'số quả':>8}{'accuracy':>11}{'macro-F1':>11}{'CE loss':>10}")
for s, n, a, f, l in rows:
    print(f"{s:<7}{n:>8}{a:>11.3f}{f:>11.3f}{l:>10.3f}")

tr, va, te = rows
print(f"\nChênh Train - Val  : accuracy {tr[2] - va[2]:+.3f} | F1 {tr[3] - va[3]:+.3f}")
print(f"Chênh Train - Test : accuracy {tr[2] - te[2]:+.3f} | F1 {tr[3] - te[3]:+.3f}")
gap = tr[3] - te[3]
if gap > 0.25:
    print("=> Overfit RÕ: Train cao hơn Test nhiều. Cần giảm độ phức tạp hoặc tăng regularization.")
elif gap > 0.10:
    print("=> Overfit NHẸ: bình thường với 132 mẫu. Early stopping và regularization đang giữ được.")
else:
    print("=> Gần như không overfit (hoặc mô hình đang underfit nếu cả hai đều thấp).")