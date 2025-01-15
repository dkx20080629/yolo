import cv2
from pathlib import Path
from datetime import datetime

class ImageSaver:
    def __init__(self, high_conf_folder="high_conf_images", low_conf_folder="low_conf_images"):
        self.image_folder = Path(high_conf_folder) / "images"
        self.label_folder = Path(high_conf_folder) / "labels"
        self.low_conf_folder = Path(low_conf_folder)
        for folder in [self.image_folder, self.label_folder, self.low_conf_folder]:
            folder.mkdir(parents=True, exist_ok=True)

    def save_image_with_label(self, img, confidence, bbox, class_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        img_path = self.image_folder / f"{timestamp}-{confidence:.2f}.jpg"
        lbl_path = self.label_folder / f"{timestamp}-{confidence:.2f}.txt"
        cv2.imwrite(str(img_path), img)
        h, w, _ = img.shape
        x1, y1, x2, y2 = bbox
        x_c, y_c, width, height = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h
        lbl_content = f"{class_id} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}"
        lbl_path.write_text(lbl_content)
        print(f"Saved: {img_path}, {lbl_path}")

    def save_low_conf_image(self, img, confidence):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filepath = self.low_conf_folder / f"{timestamp}-{confidence:.2f}.jpg"
        cv2.imwrite(str(filepath), img)
        print(f"Low-confidence image saved: {filepath}")