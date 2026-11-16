# HƯỚNG DẪN CHI TIẾT THỰC HIỆN HUẤN LUYỆN NHÁNH VISION-ONLY STUDENT (GIAI ĐOẠN 2 - CMKD)
> **Dự án:** Nhận diện quả sầu riêng chín hay không bằng Chưng cất tri thức Đa phương thức (CMKD)  
> **Tài liệu tham chiếu:** `specification/Đặc tả đề tài.pdf` (Mục 1.3.3 & Mục 1.4)  
> **Phân công phụ trách:** Leader / Member 1 (Vision Lead & Multi-Modal Framework)

---

## ❓ GIẢI ĐÁP BẢN CHẤT: GIAI ĐOẠN 2 (STUDENT DISTILLATION) HOẠT ĐỘNG THẾ NÀO?

### 1. Mô hình Student nhận đầu vào là gì?
👉 **STUDENT CHỈ NHẬN ẢNH RGB DUY NHẤT (`img` [B, 3, 384, 384]).**
- Hoàn toàn **không cần microphone** (audio) và **không cần cảm biến** (sensor) khi triển khai thực tế.
- Mục tiêu là tạo ra mô hình gọn nhẹ, chạy nhanh trên thiết bị di động/camera thực địa nhưng đạt độ chính xác tiệm cận mô hình Teacher đa giác quan.

### 2. Mô hình Teacher đóng vai trò gì trong Giai đoạn 2?
👉 **TEACHER BỊ ĐÓNG BĂNG HOÀN TOÀN (`requires_grad = False`, `eval` mode).**
- File `checkpoints/teachermodal/best_teacher.pth` đã huấn luyện ở Giai đoạn 1B được nạp lên GPU chỉ để thực hiện **Forward Pass (Inference)**.
- Khi nạp 1 batch từ `MultimodalDataset`, cùng một quả sầu riêng sẽ đưa `(img, audio, sensor)` qua Teacher để trích xuất:
  1. Vector tiềm ẩn đa giác quan $f_{mm} \in \mathbb{R}^{256}$
  2. Phân phối xác suất mềm $z_{\text{teacher}} \in \mathbb{R}^3$

### 3. Tại sao Student chỉ dùng ảnh nhưng lại phân biệt được độ chín phức tạp?
👉 Nhờ cơ chế **Căn chỉnh không gian ẩn (Latent Feature Alignment)**:
- Student có một **Projection Head** $\phi(\cdot)$ chiếu đặc trưng ảnh $f_{img}^{(S)}$ về cùng số chiều 256 của $f_{mm}$.
- Hàm mất mát $\mathcal{L}_{\text{feat}}$ ép vector ảnh của Student phải "trùng khít" với vector đa giác quan của Teacher.
- Kết quả: **Các bộ lọc tích chập của Student buộc phải tự học cách nhìn vào các gai, kẽ nứt, màu cuống để suy luận ra âm thanh tiếng gõ và độ ẩm cảm biến.**

---

## 📐 1. CÔNG THỨC TOÁN HỌC & HÀM MẤT MÁT TỔNG HỢP (MỤC 1.3.3)

Quá trình tối ưu Student được điều khiển bởi hàm mất mát 3 thành phần:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{CE}(y, \hat{y}_{\text{student}}) + \beta \mathcal{L}_{KD}(z_{\text{student}}, z_{\text{teacher}}) + \gamma \mathcal{L}_{\text{feat}}(\phi(f_{\text{img}}^{(S)}), f_{mm})$$

### 1.1. Hàm mất mát nhãn cứng ($\mathcal{L}_{CE}$ hoặc Cost-Sensitive Focal Loss)
- Đảm bảo Student bám sát nhãn thực tế $y \in \{0, 1, 2\}$.
- Khuyến nghị kế thừa **Cost-Sensitive Focal Loss** (với ma trận phạt chi phí thứ tự) đã chứng minh hiệu quả xuất sắc ở Teacher để chống nhầm lẫn giữa *Ripe* và *Overripe*:
$$\mathcal{L}_{CE} = \mathcal{L}_{\text{Cost-Sensitive-Focal}}(z_{\text{student}}, y)$$

### 1.2. Chưng cất phản hồi mềm ($\mathcal{L}_{KD}$ - Response-based Logit Matching)
- Sử dụng độ phân kỳ Kullback-Leibler (KL Divergence) với tham số nhiệt độ làm mềm $T$ (mặc định $T = 3.0$ hoặc $4.0$):
$$\mathcal{L}_{KD} = T^2 \cdot D_{KL}\left(\sigma\left(\frac{z_{\text{student}}}{T}\right) \;\parallel\; \sigma\left(\frac{z_{\text{teacher}}}{T}\right)\right)$$
- Truyền đạt "tri thức tối" (*Dark Knowledge*): giúp Student nhận biết độ chín chuyển tiếp (ví dụ: quả này có 65% Ripe, 35% Overripe).

### 1.3. Căn chỉnh biểu diễn tiềm ẩn ($\mathcal{L}_{\text{feat}}$ - Feature-based Latent Alignment)
- Chiếu vector $f_{\text{img}}^{(S)} \in \mathbb{R}^{d_S}$ qua tầng chiếu thích ứng $\phi(\cdot)$ thành $\tilde{f}_S \in \mathbb{R}^{256}$.
- Tối thiểu hóa khoảng cách Cosine hoặc sai số bình phương trung bình (MSE):
$$\mathcal{L}_{\text{feat}} = 1 - \frac{\phi(f_{\text{img}}^{(S)}) \cdot f_{mm}}{\|\phi(f_{\text{img}}^{(S)})\|_2 \; \|f_{mm}\|_2} \quad \text{hoặc} \quad \mathcal{L}_{\text{feat}} = \|\phi(f_{\text{img}}^{(S)}) - f_{mm}\|_2^2$$

> **Trọng số khuyến nghị:** $\alpha = 1.0$, $\beta = 2.0$, $\gamma = 1.0$, $T = 3.0$.

---

## 📁 2. CẤU TRÚC FOLDER VÀ CÁC FILE CẦN HIỆN THỰC

```text
Project/
├── src/
│   ├── models/
│   │   ├── student_model.py       <--- [CẦN VIẾT] Kiến trúc Vision-Only Student + Projection Head phi(.)
│   │   └── teacher_model.py       <--- [ĐÃ CÓ] Nạp best_teacher.pth (requires_grad = False)
│   ├── losses/
│   │   ├── distillation_loss.py   <--- [CẦN VIẾT] CrossModalDistillationLoss (L_CE + L_KD + L_feat)
│   │   └── classification_loss.py <--- [ĐÃ CÓ] CostSensitiveFocalLoss, compute_class_weights
│   └── datasets/
│       ├── multimodal_dataset.py  <--- [ĐÃ CÓ] Cung cấp batch (img, audio, sensor, label, fid) khi train
│       └── image_dataset.py       <--- [ĐÃ CÓ] Cung cấp chỉ img khi validate / test Student độc lập
├── workspaces/member1_vision_lead/04_train_teacher_and_student/
│   ├── train_student.py           <--- [CẦN VIẾT] Script huấn luyện Student có distillation từ Teacher
│   ├── eval_student.py            <--- [CẦN VIẾT] Script đánh giá độc lập Student (Sample & Fruit level)
│   ├── 02_train_vision_student.ipynb <--- [CẦN VIẾT] Notebook Google Colab GPU huấn luyện Student
│   └── results_student/           <--- Thư mục lưu kết quả train/test của Student
├── train_student.py               <--- [CẦN VIẾT] Root entrypoint chuyển tiếp lệnh
└── checkpoints/
    ├── teachermodal/
    │   └── best_teacher.pth       <--- Checkpoint Teacher Giai đoạn 1B (Đầu vào tri thức)
    └── studentmodal/
        └── best_student.pth       <--- Checkpoint Student Giai đoạn 2 (Đầu ra cuối cùng)
```

---

## 🛠️ 3. ĐẶC TẢ CHI TIẾT CÁC FILE SOURCE CODE CẦN HIỆN THỰC

### 3.1. File `src/models/student_model.py`
Mô hình Student chỉ nhận ảnh RGB, gồm 3 khối:
1. **Vision Backbone:**
   - Hỗ trợ kiến trúc tùy chọn: `convnextv2_tiny` (khuyến nghị cho Student siêu nhẹ ~28M tham số) hoặc `convnextv2_base` / `resnet18`.
   - Trích xuất vector đặc trưng $f_{\text{img}}^{(S)} \in \mathbb{R}^{d_S}$.
2. **Projection Head $\phi(\cdot)$:**
   - Kiến trúc: `Linear(d_S, 512) -> LayerNorm -> GELU -> Linear(512, 256) -> LayerNorm`.
   - Ánh xạ $f_{\text{img}}^{(S)}$ về cùng số chiều 256 của $f_{mm}$ để tính $\mathcal{L}_{\text{feat}}$.
3. **Classifier Head:**
   - `Linear(d_S, 3)` sinh ra logits $z_{\text{student}}$.
4. **Hàm `forward(img, return_features=True)`:**
   - Khi huấn luyện: trả về `(logits, f_img, f_proj)`.
   - Khi suy luận (Inference): chỉ cần trả về `logits` (hoặc `logits, f_proj`).

### 3.2. File `src/losses/distillation_loss.py`
Lớp `CrossModalDistillationLoss(nn.Module)`:
- Nhận vào:
  - `logits_student`: $[B, 3]$
  - `logits_teacher`: $[B, 3]$
  - `f_proj_student`: $[B, 256]$
  - `f_mm_teacher`: $[B, 256]$
  - `targets`: $[B]$
- Các tham số:
  - `alpha`: hệ số nhãn cứng (mặc định 1.0).
  - `beta`: hệ số logit distillation (mặc định 2.0).
  - `gamma`: hệ số feature alignment (mặc định 1.0).
  - `temperature`: nhiệt độ làm mềm $T$ (mặc định 3.0).
  - `feat_loss_type`: `"cosine"` (khoảng cách cosin) hoặc `"mse"` (sai số bình phương).
  - `ce_loss_type`: `"cost_sensitive_focal"` hoặc `"ce"`.
- Trả về: `loss_total, {"loss_ce": ..., "loss_kd": ..., "loss_feat": ...}`.

### 3.3. File `workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py`
Quy trình huấn luyện chưng cất:
1. **Chuẩn bị:**
   - Nạp Teacher từ `checkpoints/teachermodal/best_teacher.pth` bằng `build_teacher_from_checkpoint()`.
   - Đóng băng Teacher: `teacher.eval()`, `for p in teacher.parameters(): p.requires_grad = False`.
   - Khởi tạo Student `VisionStudent(config)` và đưa lên GPU.
2. **Vòng lặp Training (mỗi epoch):**
   - Lấy batch từ `MultimodalDataset`: `(img, audio, sensor, label, fid)`.
   - Forward Teacher (với `torch.no_grad()`):
     ```python
     with torch.no_grad():
         logits_t, f_mm_t = teacher(img, audio, sensor)
     ```
   - Forward Student:
     ```python
     logits_s, f_img_s, f_proj_s = student(img, return_features=True)
     ```
   - Tính tổng hợp 3 loss qua `CrossModalDistillationLoss`.
   - Backward & Optimizer Step trên tham số của Student (Teacher hoàn toàn không bị ảnh hưởng).
3. **Vòng lặp Validation:**
   - Chạy trên `valid_loader` **chỉ với `img`** (Student tự đứng một mình, không dùng Teacher).
   - Tổng hợp xác suất theo từng quả (`fruit_id`) bằng chiến lược `mean_prob`.
   - Đánh giá `sample_macro_f1` và `fruit_macro_f1`.
   - Lưu checkpoint tốt nhất vào `checkpoints/studentmodal/best_student.pth`.
4. **Đánh giá Test cuối cùng (1 lần):**
   - Đánh giá trên tập `test` độc lập hoàn toàn chỉ dùng ảnh RGB.
   - Xuất file so sánh chi tiết giữa Student, Teacher và Vision Unimodal.

---

## 📓 4. CẤU TRÚC NOTEBOOK GOOGLE COLAB (`02_train_vision_student.ipynb`)

Notebook được thiết kế tinh gọn gồm 6 Cells:

### 🔹 Cell 1: Mount Google Drive & Kiểm tra GPU
```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi
```

### 🔹 Cell 2: Git Pull code mới nhất từ nhánh `cocobaut`
```python
%cd /content/durian-ripeness-classification
!git fetch origin
!git checkout cocobaut
!git pull origin cocobaut
```

### 🔹 Cell 3: Cài đặt thư viện phụ thuộc
```python
!pip install -q timm torchaudio scikit-learn pandas matplotlib
```

### 🔹 Cell 4: Đồng bộ Checkpoints từ Google Drive
```python
# Đảm bảo best_teacher.pth đã có sẵn trong checkpoints/teachermodal/
!mkdir -p checkpoints/teachermodal checkpoints/studentmodal
!cp -rn /content/drive/MyDrive/checkpoints/teachermodal/* checkpoints/teachermodal/ 2>/dev/null || true
!ls -lh checkpoints/teachermodal/
```

### 🔹 Cell 5: THỰC THI HUẤN LUYỆN CHƯNG CẤT STUDENT (GIAI ĐOẠN 2)
```python
!python workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py \
    --manifest data/final/final_multimodal_manifest.csv \
    --data-dir data/final \
    --teacher-ckpt checkpoints/teachermodal/best_teacher.pth \
    --student-backbone convnextv2_tiny \
    --save-path checkpoints/studentmodal/best_student.pth \
    --results-dir workspaces/member1_vision_lead/04_train_teacher_and_student/results_student \
    --alpha 1.0 \
    --beta 2.0 \
    --gamma 1.0 \
    --temperature 3.0 \
    --batch-size 16 \
    --epochs 30 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --patience 10 \
    --select-metric fruit_macro_f1 \
    --eval-test \
    --device cuda
```

### 🔹 Cell 6: Lưu kết quả và Checkpoint Student ngược về Google Drive
```python
!cp -r checkpoints/studentmodal/* /content/drive/MyDrive/checkpoints/studentmodal/ 2>/dev/null || true
print("✅ Đã sao lưu mô hình Student thành công!")
```

---

## 🎯 5. MỤC TIÊU NGHIỆM THU ĐỒ ÁN (SUCCESS CRITERIA)

| Chỉ số đối sánh | Baseline Vision (ConvNeXt-Base) | Teacher Đa phương thức | **Kỳ vọng Student (Chỉ dùng ảnh RGB)** |
| :--- | :---: | :---: | :---: |
| **Đầu vào khi Test** | Ảnh RGB đơn lẻ | Ảnh + Âm thanh + Cảm biến | **CHỈ DÙNG ẢNH RGB** |
| **Số lượng tham số** | 88.6M (Rất nặng) | 100.4M (Đa nhánh) | **~28M (Siêu nhẹ, Tiny)** |
| **Test Sample Acc** | 40.60% | 70.63% | **62% - 68%+** *(Tăng vượt bậc so với Vision)* |
| **Test Fruit Acc** | 45.83% | 72.41% | **65% - 72%+** *(Tiệm cận Teacher)* |

👉 **Ý nghĩa khoa học khẳng định:** Student không cần cảm biến cồng kềnh hay micro thu âm ngoài vườn, chỉ cần chụp 1 bức ảnh nhưng đạt độ chính xác gần bằng hệ thống đa giác quan phức tạp nhờ phương pháp **Cross-Modal Knowledge Distillation (CMKD)**.

