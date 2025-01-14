import cv2
from pathlib import Path
import shutil


class ImageSaver:
    def __init__(self, save_folder="detected_images", low_conf_folder="low_conf_images"):
        """
        Initialize the image saver with folders for high and low-confidence images.
        """
        self.save_folder = Path(save_folder)
        self.low_conf_folder = Path(low_conf_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.low_conf_folder.mkdir(parents=True, exist_ok=True)
        self.counter = 1  # Initialize a counter for filenames

    def save_image_with_label(self, img, confidence: float, bbox: tuple[int, int, int, int], class_id: int):
        """
        Save the image with YOLO format labels.
        """
        filename = f"{confidence:.2f}-{self.counter}.jpg"
        filepath = self.save_folder / filename
        label_filepath = self.save_folder / f"{confidence:.2f}-{self.counter}.txt"

        # Save the image
        cv2.imwrite(str(filepath), img)

        # Create YOLO format label
        img_height, img_width, _ = img.shape
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        with open(label_filepath, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        print(f"Image and label saved: {filepath}, {label_filepath}")

        # Increment the counter
        self.counter += 1

    def save_low_conf_image(self, img, confidence: float):
        """
        Save the image with low confidence to a separate folder.
        """
        filename = f"{confidence:.2f}-{self.counter}.jpg"
        filepath = self.low_conf_folder / filename

        # Save the image
        cv2.imwrite(str(filepath), img)
        print(f"Low-confidence image saved: {filepath}")

        # Increment the counter
        self.counter += 1
