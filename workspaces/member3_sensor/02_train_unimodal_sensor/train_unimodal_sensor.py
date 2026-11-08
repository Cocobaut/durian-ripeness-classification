#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Huấn luyện Unimodal Baseline cho nhánh Cảm biến / Bảng / Nhiệt / Đa phổ (Tuần 2).

Phụ trách: Member 3

MÔ HÌNH ĐƯỢC CHỌN (ghi chú theo yêu cầu kế hoạch):
    SensorBackbone (MLP: Linear -> LayerNorm -> ReLU -> Dropout, hidden=(64, 48), latent=32)
    + Classification Head tuyến tính (32 -> 3). Định nghĩa trong
    workspaces/member3_sensor/01_eda_and_preprocess_sensor/preprocess_sensor.py.
    Lý do: chỉ có 132 mẫu train nên cần mạng nhỏ; LayerNorm ổn định với batch nhỏ;
    Random Forest được chạy song song làm đối chứng phi-nơ-ron.

QUY TRÌNH:
    1. Đọc data/final/{train,valid,test}/durian_characteristics_clean.csv.
    2. Trích đặc trưng nhiệt (3) và đa phổ (8) TỪ FILE THẬT trong thermal/ và multispectral/
       (các cột fg_mean_temp, band_k_mean trong manifest chỉ là chỗ trống = 0).
       Quả thiếu modality => NaN => điền median của Train + cờ mask_thermal / mask_multispectral.
    3. Fit SensorPreprocessor CHỈ trên Train, transform Val/Test.
    4. Train MLP: CE + label smoothing, AdamW (weight decay), Gaussian noise, early stopping theo
       Val Macro-F1. Chạy nhiều seed. Chọn checkpoint CHỈ dựa trên Val, Test chỉ để báo cáo.
    5. Xuất: best_sensor_unimodal.pth (backbone và head tách riêng), bảng benchmark, confusion matrix.

CÁCH DÙNG (chạy từ thư mục gốc của repo):
    python workspaces/member3_sensor/02_train_unimodal_sensor/train_unimodal_sensor.py
    python ... --feature-sets C_integrated19 A_nondestructive --seeds 42 43 44

CÁCH TUẦN 3 NẠP LẠI BACKBONE:
    ck = torch.load("checkpoints/unimodal/best_sensor_unimodal.pth", map_location="cpu")
    net = SensorBackbone(input_dim=ck["input_dim"], hidden_dims=ck["hidden_dims"],
                         latent_dim=ck["latent_dim"], num_classes=ck["num_classes"])
    net.encoder.load_state_dict(ck["backbone_state_dict"])   # chỉ backbone -> f_sensor
    # net.classifier.load_state_dict(ck["head_state_dict"])  # head (bỏ qua khi ghép Teacher)
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ------------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN & IMPORT preprocess_sensor
# ------------------------------------------------------------------------------
def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "workspaces").is_dir():
            return p
    return Path.cwd()


def import_preprocess(repo_root: Path, override: str | None):
    pre_dir = Path(override) if override else repo_root / "workspaces" / "member3_sensor" / "01_eda_and_preprocess_sensor"
    if not (pre_dir / "preprocess_sensor.py").exists():
        raise FileNotFoundError(f"Không thấy preprocess_sensor.py trong {pre_dir}")
    sys.path.insert(0, str(pre_dir))
    import preprocess_sensor as P  

    return P


# ------------------------------------------------------------------------------
# 1. TRÍCH ĐẶC TRƯNG NHIỆT & ĐA PHỔ TỪ FILE THẬT
# ------------------------------------------------------------------------------
def _fruit_id_from_name(name: str):
    m = re.search(r"durian_?0*(\d+)", name, re.IGNORECASE)
    return f"durian_{int(m.group(1)):03d}" if m else None


def collect_files(folder: Path, exts: set) -> dict:
    """Gom file theo fruit_id (tìm đệ quy, phòng trường hợp bị lồng thêm một cấp thư mục)."""
    out: dict = {}
    if not folder.is_dir():
        return out
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            fid = _fruit_id_from_name(p.name)
            if fid:
                out.setdefault(fid, []).append(p)
    return out


def thermal_vector(P, files) -> list:
    rows = []
    for f in files:
        mat = P.load_thermal_matrix(f)
        if mat is None or mat.size == 0:
            continue
        ft = P.extract_thermal_features(mat)
        rows.append([ft["fg_mean"], ft["fg_std"], ft["delta_temp_fg_bg"]])
    return list(np.mean(rows, axis=0)) if rows else [np.nan] * 3


def multispectral_vector(P, files) -> list:
    rows = []
    for f in files:
        raw = P.safe_read_tif(f)
        if raw is None:
            continue
        ft = P.extract_multispectral_features(raw)
        v = [ft[c] for c in P.MULTISPECTRAL_FEATURE_COLS]
        if all(x == 0.0 for x in v):  # định dạng không nhận ra / ảnh rỗng => coi như thiếu
            continue
        rows.append(v)
    return list(np.mean(rows, axis=0)) if rows else [np.nan] * len(P.MULTISPECTRAL_FEATURE_COLS)


def load_split(P, data_dir: Path, split: str, cache_dir: Path, rebuild: bool) -> pd.DataFrame:
    csv_path = data_dir / split / "durian_characteristics_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Không thấy {csv_path}. Bạn đã tải và đặt data/final đúng cấu trúc chưa?")
    df = pd.read_csv(csv_path)

    fid_calc = P.make_fruit_id(df)
    if "fruit_id" in df.columns and (df["fruit_id"].astype(str).values != np.array(fid_calc)).any():
        raise ValueError(f"[{split}] cột fruit_id không khớp với Replicate, kiểm tra lại CSV.")
    df["fruit_id"] = fid_calc

    cache_path = cache_dir / f"sensor_features_{split}.csv"
    if cache_path.exists() and not rebuild:
        feats = pd.read_csv(cache_path)
    else:
        th_files = collect_files(data_dir / split / "thermal", {".csv"})
        ms_files = collect_files(data_dir / split / "multispectral", {".tif", ".tiff"})
        recs = []
        for fid in df["fruit_id"]:
            rec = {"fruit_id": fid}
            rec.update(dict(zip(P.THERMAL_FEATURE_COLS, thermal_vector(P, th_files.get(fid, [])))))
            rec.update(dict(zip(P.MULTISPECTRAL_FEATURE_COLS, multispectral_vector(P, ms_files.get(fid, [])))))
            recs.append(rec)
        feats = pd.DataFrame(recs)
        cache_dir.mkdir(parents=True, exist_ok=True)
        feats.to_csv(cache_path, index=False)

    df = df.merge(feats, on="fruit_id", how="left", validate="one_to_one")
    df["mask_thermal"] = df[P.THERMAL_FEATURE_COLS[0]].notna().astype(float)
    df["mask_multispectral"] = df[P.MULTISPECTRAL_FEATURE_COLS[0]].notna().astype(float)
    return df


# ------------------------------------------------------------------------------
# 2. CÁC BỘ ĐẶC TRƯNG (FEATURE SETS)
# ------------------------------------------------------------------------------
def make_feature_sets(P) -> dict:
    phys = ["Harvest_Weight", "Horizontal_Circumference", "Vertical_Circumference", "Pedicel_Diameter"]
    lab = ["Firmness_AVE", "TSS_AVE", "Dry_Matter_AVE"]
    th, ms = list(P.THERMAL_FEATURE_COLS), list(P.MULTISPECTRAL_FEATURE_COLS)
    mth, mms = ["mask_thermal"], ["mask_multispectral"]
    return {
        # Chính thức theo thiết kế của Khải: 8 cột bảng + 3 nhiệt + 8 đa phổ (+ 2 cờ mask)
        "C_integrated19": list(P.DEFAULT_FEATURE_COLS) + th + ms + mth + mms,
        # Không dùng phép đo phá hủy (Firmness/TSS/Dry matter) và không dùng ESR_Days
        "A_nondestructive": phys + th + ms + mth + mms,
        # A + 3 phép đo lab (privileged information)
        "B_with_lab": phys + lab + th + ms + mth + mms,
        # Chỉ 8 cột bảng
        "T_tabular8": list(P.DEFAULT_FEATURE_COLS),
        # Từng nguồn riêng lẻ để đo mức đóng góp
        "H_thermal_only": th + mth,
        "M_multispectral_only": ms + mms,
    }


# ------------------------------------------------------------------------------
# 3. METRICS, PROFILING, VẼ
# ------------------------------------------------------------------------------
def compute_metrics(P, y_true, y_pred, probs) -> dict:
    labels = list(range(len(P.CLASS_ORDER)))
    pr, rc, f1c, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for i, c in enumerate(P.CLASS_ORDER):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = pr[i], rc[i], f1c[i]
    try:
        m["roc_auc_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        m["roc_auc_macro"] = float("nan")
    return {k: float(v) for k, v in m.items()}


def profile_model(model: nn.Module, in_dim: int, runs: int = 200, warmup: int = 30) -> dict:
    model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, in_dim)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return {
        "params": int(sum(p.numel() for p in model.parameters())),
        "latency_ms_mean": float(np.mean(ts)),
        "latency_ms_p95": float(np.percentile(ts, 95)),
    }


def plot_confusion(P, y_true, y_pred, title: str, path: Path) -> None:
    labels = list(range(len(P.CLASS_ORDER)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(P.CLASS_ORDER)
    ax.set_yticklabels(P.CLASS_ORDER)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title, fontsize=10)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 55 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 4. HUẤN LUYỆN MỘT LẦN CHẠY (1 feature set, 1 seed)
# ------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
    return logits.cpu()


def train_one(P, cols, seed, tr, va, te, args, device):
    set_seed(seed)
    pre = P.SensorPreprocessor(feature_cols=cols, log_cols=("Firmness_AVE",), scaler_type="standard")
    pre.fit_scaler(tr)  # CHỈ fit trên Train
    pre.check_no_overlap(va, "Val")
    pre.check_no_overlap(te, "Test")

    Xtr, ytr = pre.to_tensor(tr), pre.encode_labels(tr)
    Xva, yva = pre.to_tensor(va), pre.encode_labels(va)
    Xte, yte = pre.to_tensor(te), pre.encode_labels(te)

    counts = torch.bincount(ytr, minlength=len(P.CLASS_ORDER)).float()
    class_w = (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)  # Weighted CE (cân bằng thì = 1)
    crit_train = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)

    model = P.SensorBackbone(
        input_dim=Xtr.shape[1], hidden_dims=tuple(args.hidden), latent_dim=args.latent,
        num_classes=len(P.CLASS_ORDER), dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"f1": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    bad = 0
    n = len(Xtr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            if args.noise > 0:
                xb = xb + args.noise * torch.randn_like(xb)
            loss = crit_train(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        lv = predict(model, Xva, device)
        v_loss = F.cross_entropy(lv, yva).item()
        v_f1 = f1_score(yva.numpy(), lv.argmax(1).numpy(), average="macro", zero_division=0)
        improved = v_f1 > best["f1"] + 1e-9 or (abs(v_f1 - best["f1"]) <= 1e-9 and v_loss < best["loss"] - 1e-6)
        if improved:
            best.update(f1=v_f1, loss=v_loss, state=copy.deepcopy(model.state_dict()), epoch=ep)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best["state"])
    out = {"best_epoch": best["epoch"], "epochs_run": ep}
    for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        logits = predict(model, X, device)
        probs = F.softmax(logits, dim=-1).numpy()
        pred = probs.argmax(1)
        out[name] = compute_metrics(P, y.numpy(), pred, probs)
        out[f"{name}_pred"] = pred
        out[f"{name}_true"] = y.numpy()
    return model.cpu(), pre, out


def random_forest_baseline(P, cols, tr, va, te, seed, importance_path=None):
    pre = P.SensorPreprocessor(feature_cols=cols, log_cols=("Firmness_AVE",), scaler_type="standard").fit_scaler(tr)
    Xtr, ytr = pre.transform(tr), pre.encode_labels(tr).numpy()
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(Xtr, ytr)
    res = {}
    for name, d in (("val", va), ("test", te)):
        X, y = pre.transform(d), pre.encode_labels(d).numpy()
        probs = rf.predict_proba(X)
        res[name] = compute_metrics(P, y, probs.argmax(1), probs)
    if importance_path is not None:
        pd.DataFrame({"feature": cols, "importance": rf.feature_importances_}) \
            .sort_values("importance", ascending=False).to_csv(importance_path, index=False)
    return res


# ------------------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------------------
def parse_args(repo_root: Path):
    ap = argparse.ArgumentParser(description="Unimodal Baseline: nhánh Sensor/Thermal/Multispectral")
    ap.add_argument("--repo-root", default=str(repo_root))
    ap.add_argument("--preprocess-dir", default=None, help="Thư mục chứa preprocess_sensor.py (mặc định theo repo)")
    ap.add_argument("--data-dir", default=None, help="Mặc định <repo>/data/final")
    ap.add_argument("--out-dir", default=None,
                    help="Mặc định <repo>/workspaces/member3_sensor/02_train_unimodal_sensor/results")
    ap.add_argument("--ckpt-dir", default=None, help="Mặc định <repo>/checkpoints/unimodal")
    ap.add_argument("--feature-sets", nargs="+", default=None, help="Mặc định: chạy tất cả")
    ap.add_argument("--official-set", default="C_integrated19",
                    help="Bộ đặc trưng dùng cho best_sensor_unimodal.pth")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.05, help="Độ lệch chuẩn nhiễu Gaussian lúc train")
    ap.add_argument("--hidden", nargs="+", type=int, default=[64, 48])
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rebuild-features", action="store_true", help="Bỏ cache, trích lại đặc trưng từ file")
    return ap.parse_args()


def main():
    repo_root = find_repo_root()
    args = parse_args(repo_root)
    root = Path(args.repo_root)
    P = import_preprocess(root, args.preprocess_dir)

    data_dir = Path(args.data_dir) if args.data_dir else root / "data" / "final"
    out_dir = Path(args.out_dir) if args.out_dir else root / "workspaces" / "member3_sensor" / "02_train_unimodal_sensor" / "results"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else root / "checkpoints" / "unimodal"
    runs_dir = out_dir / "runs"
    for d in (out_dir, runs_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- Nạp dữ liệu & kiểm tra ------------------------------------------------
    print("=" * 78)
    print("[1/4] Nạp dữ liệu và trích đặc trưng nhiệt / đa phổ từ file ...")
    tr = load_split(P, data_dir, "train", out_dir, args.rebuild_features)
    va = load_split(P, data_dir, "valid", out_dir, args.rebuild_features)
    te = load_split(P, data_dir, "test", out_dir, args.rebuild_features)
    for name, d in (("train", tr), ("valid", va), ("test", te)):
        print(f"  {name:<6}: {len(d):>3} quả | có thermal: {int(d.mask_thermal.sum()):>3} | "
              f"có multispectral: {int(d.mask_multispectral.sum()):>3} | "
              f"nhãn: {d['Ripeness'].value_counts().to_dict()}")
    if tr.mask_thermal.sum() == 0 or tr.mask_multispectral.sum() == 0:
        raise RuntimeError(
            "Không trích được đặc trưng nhiệt hoặc đa phổ ở tập Train (0 quả có dữ liệu). "
            f"Kiểm tra {data_dir/'train'/'thermal'} (cần file .csv) và {data_dir/'train'/'multispectral'} "
            "(cần file .tif); tên file phải chứa 'durian_XXX'."
        )
    ids = [set(d.fruit_id) for d in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "Rò rỉ fruit_id giữa các tập!"

    fsets = make_feature_sets(P)
    chosen = args.feature_sets or list(fsets)
    if args.official_set not in chosen:
        chosen.append(args.official_set)
    for fs in chosen:
        if fs not in fsets:
            raise KeyError(f"Bộ đặc trưng không tồn tại: {fs}. Có: {list(fsets)}")
        miss = [c for c in fsets[fs] if c not in tr.columns]
        if miss:
            raise KeyError(f"[{fs}] thiếu cột trong dữ liệu: {miss}")

    # ---- Huấn luyện ------------------------------------------------------------
    print("[2/4] Huấn luyện MLP + Random Forest ...")
    run_rows, bench_rows = [], []
    best_official = None
    for fs in chosen:
        cols = fsets[fs]
        per_seed = []
        for seed in args.seeds:
            model, pre, res = train_one(P, cols, seed, tr, va, te, args, device)
            per_seed.append(res)
            row = {"feature_set": fs, "model": "MLP", "seed": seed, "n_features": len(cols),
                   "best_epoch": res["best_epoch"]}
            for split in ("val", "test"):
                for k, v in res[split].items():
                    row[f"{split}_{k}"] = v
            run_rows.append(row)
            print(f"  [{fs:<20}] seed {seed}: epoch {res['best_epoch']:>3} | "
                  f"val F1 {res['val']['macro_f1']:.3f} acc {res['val']['accuracy']:.3f} | "
                  f"test F1 {res['test']['macro_f1']:.3f} acc {res['test']['accuracy']:.3f}")
            torch.save({"full_state_dict": model.state_dict(), "feature_cols": cols},
                       runs_dir / f"sensor_{fs}_seed{seed}.pth")
            if fs == args.official_set:
                key = res["val"]["macro_f1"]  # chọn theo Val Macro-F1
                if best_official is None or key > best_official["key"]:
                    best_official = {"key": key, "model": model, "pre": pre, "res": res, "seed": seed, "cols": cols}

        def ms(split, metric):
            v = np.array([r[split][metric] for r in per_seed])
            return float(v.mean()), float(v.std())

        prof = profile_model(model, len(cols))
        bench_rows.append({
            "feature_set": fs, "model": f"MLP (mean±std, {len(args.seeds)} seeds)", "n_features": len(cols),
            "val_acc": "%.3f ± %.3f" % ms("val", "accuracy"), "val_macro_f1": "%.3f ± %.3f" % ms("val", "macro_f1"),
            "test_acc": "%.3f ± %.3f" % ms("test", "accuracy"), "test_macro_f1": "%.3f ± %.3f" % ms("test", "macro_f1"),
            "params": prof["params"], "latency_ms": round(prof["latency_ms_mean"], 4),
        })
        imp = out_dir / f"rf_feature_importance_{fs}.csv"
        rf = random_forest_baseline(P, cols, tr, va, te, seed=args.seeds[0], importance_path=imp)
        bench_rows.append({
            "feature_set": fs, "model": "RandomForest (500 trees)", "n_features": len(cols),
            "val_acc": "%.3f" % rf["val"]["accuracy"], "val_macro_f1": "%.3f" % rf["val"]["macro_f1"],
            "test_acc": "%.3f" % rf["test"]["accuracy"], "test_macro_f1": "%.3f" % rf["test"]["macro_f1"],
            "params": np.nan, "latency_ms": np.nan,
        })

    # ---- Lưu kết quả -----------------------------------------------------------
    print("[3/4] Lưu checkpoint chính thức và bảng benchmark ...")
    pre_path = out_dir / "sensor_preprocessor_unimodal.joblib"
    best_official["pre"].save(pre_path)
    ck_path = ckpt_dir / "best_sensor_unimodal.pth"
    m = best_official["model"]
    torch.save({
        "backbone_state_dict": m.encoder.state_dict(),     # f_sensor in R^latent (dùng cho Teacher Tuần 3)
        "head_state_dict": m.classifier.state_dict(),      # Classification Head tách riêng
        "full_state_dict": m.state_dict(),
        "input_dim": len(best_official["cols"]), "hidden_dims": list(args.hidden),
        "latent_dim": args.latent, "num_classes": len(P.CLASS_ORDER), "dropout": args.dropout,
        "feature_cols": best_official["cols"], "feature_set": args.official_set,
        "class_order": list(P.CLASS_ORDER), "seed": best_official["seed"],
        "val_macro_f1": best_official["res"]["val"]["macro_f1"],
        "test_macro_f1": best_official["res"]["test"]["macro_f1"],
        "preprocessor_path": str(pre_path),
    }, ck_path)

    pd.DataFrame(run_rows).to_csv(out_dir / "sensor_unimodal_runs.csv", index=False)
    bench = pd.DataFrame(bench_rows)
    bench.to_csv(out_dir / "sensor_unimodal_benchmark.csv", index=False)
    (out_dir / "sensor_unimodal_benchmark.md").write_text(df_to_markdown(bench), encoding="utf-8")
    plot_confusion(P, best_official["res"]["test_true"], best_official["res"]["test_pred"],
                   f"Sensor ({args.official_set}) - Test", out_dir / "confusion_matrix_sensor_test.png")
    plot_confusion(P, best_official["res"]["val_true"], best_official["res"]["val_pred"],
                   f"Sensor ({args.official_set}) - Val", out_dir / "confusion_matrix_sensor_val.png")
    (out_dir / "official_summary.json").write_text(json.dumps({
        "feature_set": args.official_set, "seed": best_official["seed"], "checkpoint": str(ck_path),
        "val": best_official["res"]["val"], "test": best_official["res"]["test"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[4/4] Bảng benchmark:")
    print(bench.to_string(index=False))
    print("-" * 78)
    print(f"Checkpoint chính thức : {ck_path}")
    print(f"Bộ đặc trưng          : {args.official_set} (seed {best_official['seed']}, chọn theo Val Macro-F1)")
    print(f"Kết quả & biểu đồ     : {out_dir}")


if __name__ == "__main__":
    main()