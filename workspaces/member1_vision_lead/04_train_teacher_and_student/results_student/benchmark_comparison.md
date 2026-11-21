# CMKD Performance Benchmark Comparison (Test Set)

| Model | Test Input | Sample Acc | Sample Macro-F1 | Fruit Acc | Fruit Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Vision Unimodal (ConvNeXtV2-Base Multi-View)** | Multi-View RGB | 38.26% | 38.05% | 58.33% | 55.59% |
| **2. Sensor Unimodal (Integrated MLP)** | Sensor (19 fts) | 68.97% | 67.91% | 68.97% | 67.91% |
| **3. Audio Unimodal (ResNet-18, 20 fruits)** | Audio Only | 84.21% | 81.37% | 95.00% | 93.73% |
| **4. Multi-Modal Teacher (Cross-Attention)** | Multi-View RGB + Audio + Sensor | 80.85% | 78.21% | 79.31% | 78.41% |
| **4b. Multi-Modal Teacher (same 24 RGB fruits as Student)** | Multi-View RGB + Audio + Sensor | — | — | 79.17% | 79.08% |
| **5. Vision-Only Student (CMKD Multi-View)** | Multi-View RGB Only | 50.00% | 48.68% | 50.00% | 48.68% |
