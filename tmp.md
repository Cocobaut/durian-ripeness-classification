# HƯỚNG DẪN CHI TIẾT THỰC HIỆN HUẤN LUYỆN NHÁNH MULTI-MODAL TEACHER (GIAI ĐOẠN 1B)
> **Dự án:** Nhận diện quả sầu riêng chín hay không bằng Chưng cất tri thức Đa phương thức (CMKD)  
> **Tài liệu tham chiếu:** `specification/Đặc tả đề tài.pdf` (Mục 1.3.1, 1.3.2 & Mục 1.4)  
> **Phân công phụ trách:** Leader / Member 1 (Vision Lead & Multi-Modal Framework)

---

## ❓ GIẢI ĐÁP QUAN TRỌNG: KHI TRAIN TEACHER CÓ CẬP NHẬT LẠI CÁC BACKBONE KHÔNG?

👉 **CÂU TRẢ LỜI: CÓ CẬP NHẬT TRỌNG SỐ TRONG MÔ HÌNH (Ở PHA 2), NHƯNG TUYỆT ĐỐI KHÔNG GHI ĐÈ LÊN CÁC FILE `.pth` GỐC CỦA 3 NHÁNH!**

Cụ thể theo cơ chế Mục 1.3.2 trong file Đặc tả:
1. **Ở Pha 1 (Warm-up Freeze - 5 epoch đầu):**  
   - 3 Backbone được **đóng băng hoàn toàn** (`requires_grad = False`). Trọng số của chúng được giữ nguyên 100% từ 3 checkpoint unimodal nhằm ổn định khối Hợp nhất (Fusion Module), chống hiện tượng lấn át phương thức (*Modality Dominance*).
2. **Ở Pha 2 (Joint Fine-tuning with Differential LRs - các epoch tiếp theo):**  
   - Cả 3 Backbone được **mở khóa** (`requires_grad = True`) và **được cập nhật trọng số cùng lúc với Fusion Module**.
   - Tuy nhiên, chúng được học với tốc độ học cực nhỏ ($\text{LR} = 10^{-5}$) trong khi Fusion học với $\text{LR} = 5 \cdot 10^{-4}$ đến $10^{-3}$. Việc này giúp các backbone tinh chỉnh nhẹ để thích ứng tối đa với không gian đa phương thức.
3. **Quy tắc bảo toàn File Checkpoint:**  
   - **Các file checkpoint unimodal gốc** trong folder `unimodal/` (`vision/`, `audio/`, `sensor/`) **ĐƯỢC GIỮ NGUYÊN**, không bị ghi đè, để dùng làm mốc đối chuẩn thực nghiệm (*Unimodal Baselines* so sánh ở Mục 1.4).
   - Toàn bộ trọng số mới (gồm cả 3 Backbone đã tinh chỉnh + Khối Fusion + Classifier Head) sẽ được **lưu tích hợp chung vào một file checkpoint duy nhất của Teacher**:
     👉 Lưu tại folder con: `teachermodal/best_teacher.pth`.

---

## 📁 1. THỰC HIỆN Ở FOLDER NÀO? (NƠI CODE & QUẢN LÝ FILE)

### Khu vực 1: Không gian nghiên cứu & thực nghiệm (Workspace)
👉 **Thư mục:** `workspaces/member1_vision_lead/04_train_teacher_and_student/`
- **Mục đích:** Nơi Member 1 lưu trữ Notebook chạy Colab và script huấn luyện chính.
- **Các file trong folder này:**
  1. `01_train_multimodal_teacher.ipynb`: Notebook chạy trên Google Colab GPU (L4/T4/A100).
     > **LƯU Ý:** Notebook này được tối giản tối đa: **Chỉ dùng để Git Pull code mới nhất từ GitHub, kết nối nạp Checkpoints/Dataset từ Google Drive, và kích hoạt lệnh chạy file `train_teacher.py`**.
  2. `train_teacher.py`: Script Python chứa toàn bộ logic huấn luyện 2 pha, nạp dữ liệu và xuất kết quả.
  3. `results/`: Lưu logs huấn luyện, biểu đồ loss/F1 curves, confusion matrix.

### Khu vực 2: Thư viện lõi dùng chung của toàn dự án (Core Source Code)
👉 **Thư mục:** `src/` và file chạy root `train_teacher.py`
- `src/losses/classification_loss.py`: **[CÓ DÙNG]** Weighted Cross-Entropy Loss + Label Smoothing cho Teacher.
- `src/datasets/image_dataset.py`, `audio_dataset.py`, `sensor_dataset.py`: **[CÓ DÙNG]** 3 Building Blocks xử lý dữ liệu cho từng kênh.
- `src/datasets/multimodal_dataset.py`: **[CỐT LÕI]** Ghép 3 kênh thành tuple 5 thành phần `(img, audio, sensor, label, fruit_id)`.
- `src/models/teacher_model.py`: **[CỐT LÕI]** Kiến trúc mô hình Teacher (ConvNeXt V2 + ResNet-18 + SensorMLP + Fusion + Head).
- `train_teacher.py` (ở thư mục gốc): Wrapper gọi huấn luyện.

---

## 🚀 2. CHẠY TRÊN FOLDER NÀO? (EXECUTION DIRECTORY)

### ⚠️ QUY TẮC BẮT BUỘC: LUÔN CHẠY TỪ THƯ MỤC GỐC DỰ ÁN (PROJECT ROOT)
```text
E:\Hcmut material\Artificial_Intelligence\Computer_Vision\Project
```
*(Trên Colab: `cd /content/durian-ripeness-cmkd` hoặc thư mục repo sau khi `git clone`).*

---

## 🌐 3. LIÊN KẾT GOOGLE DRIVE VÀ CẤU TRÚC THƯ MỤC CHECKPOINTS

* 🔗 **Google Drive Checkpoints Folder:**  
  👉 [Truy cập Google Drive Checkpoints](https://drive.google.com/drive/folders/1K6II5aFG31kYrl7rsJj53u2Np4ZzjzG_?usp=drive_link)

### Cấu trúc tổ chức thư mục trên Google Drive & Local:
```text
checkpoints/ (hoặc MyDrive/.../checkpoints/)
│
├── unimodal/                                    <--- [CHECKPOINT ĐẦU VÀO ĐÃ CÓ SẴN TRÊN DRIVE]
│   ├── vision/ (hoặc image/)                    <--- best_vision_unimodal.pth (ConvNeXt V2-Base, 1024-dim)
│   ├── audio/                                   <--- audio_backbone_teacher.pth (ResNet-18 log-Mel, 512-dim)
│   └── sensor/                                  <--- best_sensor_unimodal.pth (SensorMLP, 32-dim)
│
└── teachermodal/                                <--- [FOLDER MỚI: LƯU CHECKPOINT TEACHER SAU TRAIN]
    └── best_teacher.pth                         <--- Checkpoint chứa cả 3 backbones + Fusion + Head
```

---

## 📓 4. NỘI DUNG CHI TIẾT CHO NOTEBOOK GOOGLE COLAB
*(File: `workspaces/member1_vision_lead/04_train_teacher_and_student/01_train_multimodal_teacher.ipynb`)*

Theo đúng yêu cầu của bạn, Notebook được tinh gọn: **Không viết lại code model trong notebook**, mà chỉ đóng vai trò Runner gồm 5 ô lệnh (Cells):

### 🔹 Cell 1: Kết nối Google Drive & Kiểm tra GPU
```python
# ==============================================================================
# 1. KẾT NỐI GOOGLE DRIVE & KIỂM TRA GPU
# ==============================================================================
import os
import sys
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
print("✅ Đã kết nối Google Drive thành công!")

# Kiểm tra GPU được cấp phát (khuyến nghị GPU L4 hoặc A100)
!nvidia-smi
```

### 🔹 Cell 2: Git Clone hoặc Git Pull code mới nhất từ GitHub
```python
# ==============================================================================
# 2. ĐỒNG BỘ CODE MỚI NHẤT TỪ GITHUB
# ==============================================================================
%cd /content

REPO_URL = "https://github.com/Cocobaut/durian-ripeness-cmkd.git"  # Đổi URL repo nếu cần
REPO_DIR = "/content/durian-ripeness-cmkd"

if not os.path.exists(REPO_DIR):
    !git clone $REPO_URL
    %cd $REPO_DIR
    !git checkout cocobaut
else:
    %cd $REPO_DIR
    !git pull origin cocobaut

print(f"✅ Đang làm việc tại thư mục gốc: {os.getcwd()}")
```

### 🔹 Cell 3: Cài đặt thư viện phụ thuộc siêu tốc với `uv`
```python
# ==============================================================================
# 3. CÀI ĐẶT THƯ VIỆN PHỤ THUỘC
# ==============================================================================
!pip install -q uv
!uv pip install --system -r requirements.txt
print("✅ Đã hoàn tất cài đặt toàn bộ thư viện!")
```

### 🔹 Cell 4: Thiết lập liên kết dữ liệu & checkpoints từ Google Drive
```python
# ==============================================================================
# 4. THIẾT LẬP ĐƯỜNG DẪN CHECKPOINTS & DATASET TỪ GOOGLE DRIVE
# ==============================================================================
from pathlib import Path

# Đường dẫn folder trên Google Drive của bạn
# (Hãy điều chỉnh tên thư mục gốc trên Drive cho đúng vị trí bạn lưu)
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/durian-ripeness-cmkd")  # Hoặc đường dẫn Drive của bạn

DRIVE_CHECKPOINTS = DRIVE_PROJECT_ROOT / "checkpoints"
DRIVE_DATA = DRIVE_PROJECT_ROOT / "data" / "final"

# Tạo folder teachermodal trên Google Drive nếu chưa có
(DRIVE_CHECKPOINTS / "teachermodal").mkdir(parents=True, exist_ok=True)

# Tạo liên kết tượng trưng (Symlink) để code đọc mượt mà từ repo
!mkdir -p checkpoints data/final
!ln -sfn "{str(DRIVE_CHECKPOINTS)}/unimodal" checkpoints/unimodal
!ln -sfn "{str(DRIVE_CHECKPOINTS)}/teachermodal" checkpoints/teachermodal
!ln -sfn "{str(DRIVE_DATA)}"/* data/final/

print("✅ Đã tạo liên kết dữ liệu và checkpoint thành công:")
print(" - Checkpoint Unimodal:", os.path.exists("checkpoints/unimodal"))
print(" - Thư mục lưu Teacher:", os.path.exists("checkpoints/teachermodal"))
print(" - Dữ liệu Data Final:", os.path.exists("data/final/final_multimodal_manifest.csv"))
```

### 🔹 Cell 5: Kích hoạt chạy script `train_teacher.py`
```python
# ==============================================================================
# 5. THỰC THI HUẤN LUYỆN MULTI-MODAL TEACHER
# ==============================================================================
# Chạy trực tiếp từ Root Repo, lưu checkpoint vào folder teachermodal
!python workspaces/member1_vision_lead/04_train_teacher_and_student/train_teacher.py \
    --manifest data/final/final_multimodal_manifest.csv \
    --data-dir data/final \
    --vision-ckpt checkpoints/unimodal/vision/best_vision_unimodal.pth \
    --audio-ckpt checkpoints/unimodal/audio/audio_backbone_teacher.pth \
    --sensor-ckpt checkpoints/unimodal/sensor/best_sensor_unimodal.pth \
    --save-path checkpoints/teachermodal/best_teacher.pth \
    --batch-size 8 \
    --epochs 30 \
    --warmup-epochs 5 \
    --device cuda
```

---

## 🛠️ 5. SCRIPT HUẤN LUYỆN `train_teacher.py` HOÀN CHỈNH

Đặt tại: `workspaces/member1_vision_lead/04_train_teacher_and_student/train_teacher.py`  
*(Cũng có thể gọi qua file wrapper tại thư mục gốc `train_teacher.py`)*:

```python
"""Script huấn luyện Multi-Modal Teacher Model (2-Phase Training).

Phụ trách: Leader / Member 1
Mục tiêu:
- Tinh chỉnh đồng thời 3 Backbone + Fusion Module
- Checkpoint đầu vào: nạp từ checkpoints/unimodal/{vision, audio, sensor}
- Checkpoint đầu ra: lưu vào checkpoints/teachermodal/best_teacher.pth
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# Thiết lập đường dẫn import src
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.datasets.multimodal_dataset import MultimodalDataset
from src.models.teacher_model import MultiModalTeacher
from src.losses.classification_loss import DurianClassificationLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Multi-Modal Teacher Model")
    parser.add_argument("--manifest", type=str, default="data/final/final_multimodal_manifest.csv")
    parser.add_argument("--data-dir", type=str, default="data/final")
    parser.add_argument("--vision-ckpt", type=str, default="checkpoints/unimodal/vision/best_vision_unimodal.pth")
    parser.add_argument("--audio-ckpt", type=str, default="checkpoints/unimodal/audio/audio_backbone_teacher.pth")
    parser.add_argument("--sensor-ckpt", type=str, default="checkpoints/unimodal/sensor/best_sensor_unimodal.pth")
    parser.add_argument("--save-path", type=str, default="checkpoints/teachermodal/best_teacher.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MULTI-MODAL TEACHER (GIAI ĐOẠN 1B)")
    print(f"[*] Thiết bị tính toán: {device}")
    print(f"[*] Vision ckpt   : {args.vision_ckpt}")
    print(f"[*] Audio ckpt    : {args.audio_ckpt}")
    print(f"[*] Sensor ckpt   : {args.sensor_ckpt}")
    print(f"[*] Checkpoint lưu: {args.save_path}")
    print("=" * 70)

    # Đảm bảo folder đích teachermodal tồn tại
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 1. Dataset & DataLoader
    train_dataset = MultimodalDataset(args.manifest, args.data_dir, split="train", is_training=True)
    val_dataset = MultimodalDataset(args.manifest, args.data_dir, split="valid", is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Khởi tạo Mô hình & Loss Function
    model = MultiModalTeacher(
        vision_ckpt=args.vision_ckpt,
        audio_ckpt=args.audio_ckpt,
        sensor_ckpt=args.sensor_ckpt
    ).to(device)

    criterion = DurianClassificationLoss(label_smoothing=0.05).to(device)
    best_val_f1 = 0.0

    # --------------------------------------------------------------------------
    # PHA 1: WARM-UP FREEZE (5 Epochs đầu)
    # --------------------------------------------------------------------------
    print(f"\n>>> [PHA 1] WARM-UP FREEZE ({args.warmup_epochs} EPOCHS): Đóng băng 3 Backbone, chỉ tối ưu Fusion...")
    model.freeze_backbones()
    optimizer_p1 = torch.optim.AdamW([
        {"params": model.fusion.parameters(), "lr": 1e-3, "weight_decay": 1e-2},
        {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-2},
    ])

    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        for img, audio, sensor, label, _ in tqdm(train_loader, desc=f"Pha 1 - Epoch {epoch}/{args.warmup_epochs}"):
            img, audio, sensor, label = img.to(device), audio.to(device), sensor.to(device), label.to(device)
            optimizer_p1.zero_grad()
            logits, _ = model(img, audio, sensor)
            loss = criterion(logits, label)
            loss.backward()
            optimizer_p1.step()
            total_loss += loss.item()
        print(f"Pha 1 - Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

    # --------------------------------------------------------------------------
    # PHA 2: JOINT FINE-TUNING (Epoch tiếp theo -> Hết)
    # --------------------------------------------------------------------------
    print(f"\n>>> [PHA 2] JOINT FINE-TUNING ({args.epochs - args.warmup_epochs} EPOCHS): Mở khóa toàn mạng với Differential LR...")
    model.unfreeze_all()
    optimizer_p2 = torch.optim.AdamW([
        {"params": model.vision_backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
        {"params": model.audio_backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
        {"params": model.sensor_backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
        {"params": model.fusion.parameters(), "lr": 5e-4, "weight_decay": 1e-2},
        {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)

    for epoch in range(args.warmup_epochs + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for img, audio, sensor, label, _ in tqdm(train_loader, desc=f"Pha 2 - Epoch {epoch}/{args.epochs}"):
            img, audio, sensor, label = img.to(device), audio.to(device), sensor.to(device), label.to(device)
            optimizer_p2.zero_grad()
            logits, _ = model(img, audio, sensor)
            loss = criterion(logits, label)
            loss.backward()
            optimizer_p2.step()
            total_loss += loss.item()
        scheduler.step()

        # Đánh giá tập Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for img, audio, sensor, label, _ in val_loader:
                img, audio, sensor = img.to(device), audio.to(device), sensor.to(device)
                logits, _ = model(img, audio, sensor)
                preds = logits.argmax(dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(label.tolist())

        val_f1 = f1_score(val_targets, val_preds, average="macro")
        print(f"[*] Epoch {epoch} - Val Loss: {total_loss / len(train_loader):.4f} | Val Macro-F1: {val_f1:.4f}")

        # Lưu checkpoint Teacher tốt nhất vào folder teachermodal
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "feature_dim_mm": 256,
                "class_order": ["Unripe", "Ripe", "Overripe"]
            }, args.save_path)
            print(f"🔥 ĐÃ LƯU BEST TEACHER CHECKPOINT TẠI: {args.save_path} (F1: {best_val_f1:.4f})")

    print("\n✅ HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN TEACHER MODEL!")

if __name__ == "__main__":
    main()
```

---

## 📊 6. ĐÁNH GIÁ & KIỂM THỬ KẾT QUẢ VỚI SCRIPT MEMBER 2

Sau khi có file `checkpoints/teachermodal/best_teacher.pth`, chạy lệnh kiểm thử độc lập:
```powershell
python workspaces/member2_audio/03_test_eval_teacher_and_gradcam/eval_teacher.py `
    --checkpoint checkpoints/teachermodal/best_teacher.pth `
    --manifest data/final/final_multimodal_manifest.csv `
    --test-dir data/final/test `
    --input-mode multimodal `
    --output-dir workspaces/member1_vision_lead/04_train_teacher_and_student/results
```

---

## 🎯 7. CHECKLIST NGHIỆM THU

| Hạng mục | Đường dẫn kiểm tra | Trạng thái đạt chuẩn |
| :--- | :--- | :--- |
| **Checkpoints đầu vào** | `checkpoints/unimodal/{vision, audio, sensor}/` | Đã tải từ Google Drive vào đúng folder. |
| **Cập nhật Backbone** | In-memory ở Pha 2 | 3 backbone được tinh chỉnh cùng fusion, **không ghi đè file unimodal cũ**. |
| **Checkpoint đầu ra** | `checkpoints/teachermodal/best_teacher.pth` | Đã lưu mô hình Teacher hoàn chỉnh vào folder `teachermodal`. |
| **Colab Notebook** | `workspaces/member1_vision_lead/04_train_teacher_and_student/01_train_multimodal_teacher.ipynb` | Chỉ thực hiện: git pull, load Drive, và chạy lệnh `python train_teacher.py`. |
| **Benchmark Test** | `workspaces/member1_vision_lead/04_train_teacher_and_student/results/` | Báo cáo Upper Bound F1-score & Modality Ablation (w/o audio, w/o sensor). |
