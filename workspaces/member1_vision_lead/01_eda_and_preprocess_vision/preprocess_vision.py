"""Module tiền xử lý ảnh (Vision Modality).
Phụ trách: Member 1 (Vision Lead)

Mục tiêu:
- Định nghĩa class VisionPreprocessor để chuẩn hóa ảnh RGB quả sầu riêng.
- Member 1 chỉ cần gọi class này ở bước '03_create_final_dataset' và khi nạp mô hình
"""

# import torchvision.transforms as transforms
# from PIL import Image
# import torch


class VisionPreprocessor:
    """Class tiền xử lý cho nhánh ảnh thị giác (Vision Modality)."""

    def __init__(self, image_size=(224, 224)):
        # TODO: Khởi tạo kích thước ảnh mục tiêu và các tham số chuẩn hóa (ImageNet mean/std)
        self.image_size = image_size
        pass
