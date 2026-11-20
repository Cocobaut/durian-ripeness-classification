# Bảng đối sánh hiệu năng CMKD (Test Set)

| Mô hình | Đầu vào khi Test | Sample Acc | Sample Macro-F1 | Fruit Acc | Fruit Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Vision Unimodal (ConvNeXt-Base)** | RGB Only | 40.60% | 41.00% | 45.83% | 43.51% |
| **2. Sensor Unimodal (Integrated MLP)** | Sensor (19 fts) | 68.97% | 67.91% | 68.97% | 67.91% |
| **3. Audio Unimodal (ResNet-18, 20 fruits)** | Audio Only | 84.21% | 81.37% | 95.00% | 93.73% |
| **4. Multi-Modal Teacher (Cross-Attention)** | RGB + Audio + Sensor | 70.63% | 74.39% | 72.41% | 70.30% |
| **5. Vision-Only Student (CMKD Distilled)** | RGB Only | 49.66% | 49.55% | 58.33% | 57.72% |
