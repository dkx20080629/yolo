import cv2
from pathlib import Path
from datetime import datetime

class ImageSaver:
    def __init__(self, high_conf_folder="high_conf_images", low_conf_folder="low_conf_images"):
        """
        Initialize the image saver with folders for high and low-confidence images.
        """
        self.high_conf_folder = Path(high_conf_folder)
        self.low_conf_folder = Path(low_conf_folder)

        # Create subdirectories for images and labels within high_conf_folder
        self.image_folder = self.high_conf_folder / "images"
        self.label_folder = self.high_conf_folder / "labels"

        # Create the necessary directories
        self.image_folder.mkdir(parents=True, exist_ok=True)
        self.label_folder.mkdir(parents=True, exist_ok=True)
        self.low_conf_folder.mkdir(parents=True, exist_ok=True)

    def save_image_with_label(self, img, confidence: float, bbox: tuple[int, int, int, int], class_id: int):
        """
        Save the image with YOLO format labels in high_conf_folder.
        """
        # Get current system time as the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        image_filename = f"{timestamp}-{confidence:.2f}.jpg"
        label_filename = f"{timestamp}-{confidence:.2f}.txt"

        image_filepath = self.image_folder / image_filename
        label_filepath = self.label_folder / label_filename

        # Save the image
        cv2.imwrite(str(image_filepath), img)

        # Create YOLO format label
        img_height, img_width, _ = img.shape
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        with open(label_filepath, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        print(f"Image and label saved: {image_filepath}, {label_filepath}")

    def save_low_conf_image(self, img, confidence: float):
        """
        Save the image with low confidence to the low_conf_folder.
        """
        # Get current system time as the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = f"{timestamp}-{confidence:.2f}.jpg"
        filepath = self.low_conf_folder / filename

        # Save the image
        cv2.imwrite(str(filepath), img)
        print(f"Low-confidence image saved: {filepath}")
