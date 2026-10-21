# Durian Ripeness Cross-Modal Knowledge Distillation (CMKD)

Dự án phân loại độ chín sầu riêng ứng dụng kỹ thuật **Chưng cất tri thức Đa phương thức (Cross-Modal Knowledge Distillation)**:
- **Teacher Model**: Hợp nhất đa phương thức (Ảnh thị giác + Âm thanh gõ + Cảm biến/Bảng số liệu).
- **Student Model**: Nhánh ảnh thị giác (Vision-Only) nhưng tiếp thu tri thức âm thanh và cảm biến từ Teacher, phục vụ triển khai thực tế nhẹ và chính xác chỉ với camera.

---

## 🛠️ Hướng dẫn cài đặt môi trường (sử dụng `uv`)

Dự án khuyến khích sử dụng **`uv`** (trình quản lý gói Python siêu tốc) để đồng bộ môi trường nhanh chóng:

### 1. Cài đặt `uv` (nếu máy chưa có)
* **Trên Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  # Hoặc cài qua pip:
  pip install uv
  ```
* **Trên Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Kích hoạt môi trường ảo & Cài đặt thư viện
* **Nếu đã có sẵn môi trường `uv` trong dự án:**
  ```bash
  uv sync
  ```
* **Hoặc tự kích hoạt môi trường ảo và cài đặt qua `requirements.txt`:**
  * **Trên Windows:**
    ```powershell
    .venv\Scripts\activate
    # hoặc: venv\Scripts\activate

    uv pip install -r requirements.txt
    ```
  * **Trên Linux / macOS:**
    ```bash
    source .venv/bin/activate
    # hoặc: source venv/bin/activate

    uv pip install -r requirements.txt
    ```

---

## 📦 Hướng dẫn tải Dataset từ Google Drive & Cấu trúc dữ liệu

### 1. Danh sách liên kết Google Drive tải Dataset

Thành viên tự truy cập vào các đường dẫn Google Drive công khai dưới đây để tải dữ liệu về máy:

* 📥 **Thư mục Dữ liệu gốc (`data/raw/`):**  
  👉 [Google Drive - data/raw](https://drive.google.com/drive/folders/1gIhGj-txv1ZF_RfslbAh7bagNZdnIRhL?usp=drive_link)
* 📥 **Thư mục Dữ liệu đã chia sẵn (`data/split_data/`):**  
  👉 [Google Drive - data/split_data](https://drive.google.com/drive/folders/1ro-ETqoDNCuDwbYdZkvDuztLAF5B0XTC?usp=drive_link)
  * 📂 **Tập Train:** [Google Drive - split_data/train](https://drive.google.com/drive/folders/18jp-lYZM4WorLQxH2CHGdPBjyMOYjAJG?usp=drive_link)
  * 📂 **Tập Val:** [Google Drive - split_data/val](https://drive.google.com/drive/folders/11L5txIHnHls-vJzTiVNLYhlzNJMvGazp?usp=drive_link)
  * 📂 **Tập Test:** [Google Drive - split_data/test](https://drive.google.com/drive/folders/1GBJ9aoDJ_ZmgnHg2inM8RtAjRS3-wKZ0?usp=drive_link)

> **Lưu ý:** Sau khi tải về máy cá nhân, bạn chỉ cần bố trí và giải nén đúng theo cấu trúc thư mục chuẩn dưới đây để toàn bộ mã nguồn có thể tự động nhận diện dữ liệu.

---

### 2. Cấu trúc thư mục Dataset chuẩn trên máy cá nhân (Local)

Dữ liệu được tổ chức thống nhất trong thư mục `data/` của dự án:

```text
data/
├── raw/                                         # Dữ liệu gốc thu thập từ 189 quả sầu riêng
│   ├── dataset_clean_multispectral/             # Ảnh phổ đa kênh GeoTIFF (VD như một số file dưới đây)
│   │   ├── durian_001_Immature_A_Unripe_COMBINED.tif
│   │   ├── durian_002_Immature_A_Unripe_COMBINED.tif
│   │   └── ...
│   ├── dataset_clean_rgb/                       # Ảnh thị giác RGB chụp 6 góc (VD như một số file dưới đây)
│   │   ├── durian_001_Immature_A_Unripe_TOP.jpg
│   │   ├── durian_001_Immature_A_Unripe_BOT.jpg
│   │   ├── durian_001_Immature_A_Unripe_SA.jpg
│   │   └── ...
│   ├── dataset_clean_sound/                     # Âm thanh gõ sầu riêng (VD như một số file dưới đây)
│   │   ├── IM_CA_UN_1.wav                       # File ghi âm bằng điện thoại Android
│   │   ├── 1-U.m4a                              # File ghi âm bằng điện thoại iPhone
│   │   └── ...
│   ├── dataset_clean_thermal/                   # Ảnh nhiệt & ma trận nhiệt (VD như một số file dưới đây)
│   │   ├── durian_001_Immature_A_Unripe_TOP_thermal.jpg
│   │   ├── durian_001_Immature_A_Unripe_TOP_thermal.csv
│   │   └── ...
│   ├── durian_characteristics_cleaned.csv       # Bảng đặc tính cảm biến & nhãn 189 quả (Firmness_AVE, TSS_AVE, Dry_Matter_AVE, ...)
│   └── README.md                                # Tài liệu kỹ thuật chi tiết của dataset
│
└── split_data/                                  # Dữ liệu đã phân chia nhất quán theo từng quả
    ├── sample_split_mapping.csv                 # Bảng tra cứu toàn bộ 189 quả thuộc tập nào (train/val/test)
    │
    ├── train/                                   # Tập huấn luyện (132 quả ~ 70%)
    │   ├── dataset_clean_rgb/                   # (VD như một số file dưới đây)
    │   │   ├── durian_002_Immature_A_Unripe_TOP.jpg
    │   │   └── ...
    │   ├── dataset_clean_sound/                 # (VD như một số file dưới đây)
    │   │   ├── IM_CA_UN_2.wav
    │   │   └── ...
    │   ├── dataset_clean_thermal/               # (VD như một số file dưới đây)
    │   │   ├── durian_002_Immature_A_Unripe_TOP_thermal.jpg
    │   │   └── ...
    │   ├── dataset_clean_multispectral/         # (VD như một số file dưới đây)
    │   │   ├── durian_002_Immature_A_Unripe_COMBINED.tif
    │   │   └── ...
    │   └── durian_characteristics_cleaned.csv   # Bảng đặc tính 132 dòng của tập Train
    │
    ├── val/                                     # Tập kiểm định (28 quả ~ 15%)
    │   ├── dataset_clean_rgb/                   # (VD như một số file dưới đây)
    │   │   ├── durian_001_Immature_A_Unripe_TOP.jpg
    │   │   └── ...
    │   ├── dataset_clean_sound/                 # (VD như một số file dưới đây)
    │   │   ├── IM_CA_UN_1.wav
    │   │   └── ...
    │   ├── dataset_clean_thermal/               # (VD như một số file dưới đây)
    │   │   ├── durian_001_Immature_A_Unripe_TOP_thermal.jpg
    │   │   └── ...
    │   ├── dataset_clean_multispectral/         # (VD như một số file dưới đây)
    │   │   ├── durian_001_Immature_A_Unripe_COMBINED.tif
    │   │   └── ...
    │   └── durian_characteristics_cleaned.csv   # Bảng đặc tính 28 dòng của tập Val
    │
    └── test/                                    # Tập kiểm thử đánh giá (29 quả ~ 15%)
        ├── dataset_clean_rgb/                   # (VD như một số file dưới đây)
        │   ├── durian_004_Immature_A_Unripe_TOP.jpg
        │   └── ...
        ├── dataset_clean_sound/                 # (VD như một số file dưới đây)
        │   ├── IM_CA_UN_4.wav
        │   └── ...
        ├── dataset_clean_thermal/               # (VD như một số file dưới đây)
        │   ├── durian_004_Immature_A_Unripe_TOP_thermal.jpg
        │   └── ...
        ├── dataset_clean_multispectral/         # (VD như một số file dưới đây)
        │   ├── durian_004_Immature_A_Unripe_COMBINED.tif
        │   └── ...
        └── durian_characteristics_cleaned.csv   # Bảng đặc tính 29 dòng của tập Test
```

---

### 3. Đảm bảo tính nhất quán Đa phương thức (Cross-Modal Consistency)

* **Phân chia theo Quả (Sample-Level):** Tất cả các góc ảnh RGB, file âm thanh, ảnh nhiệt, ảnh phổ và dòng CSV của cùng một quả sầu riêng **bắt buộc phải cùng nằm trong một tập duy nhất** (`train`, `val`, hoặc `test`).
* **Không bị Data Leakage:** Tránh tình trạng ảnh góc TOP của quả nằm ở `train` mà bảng CSV hoặc ảnh góc BOT lại nằm ở `test`.
* **Phân tầng cân bằng (Stratified):** Đảm bảo tỷ lệ 3 lớp độ chín (`Unripe`, `Ripe`, `Overripe`) được phân bổ đồng đều ở cả 3 tập Train (44 quả/lớp), Val (~9-10 quả/lớp), Test (~9-10 quả/lớp).