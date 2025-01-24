from pathlib import Path
from ultralytics import YOLO


def auto_annotate_bboxes_yolo_format(
    data,
    det_model="note2025-latest.engine",
    device=0,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    max_det=300,
    classes=None,
    output_dir=None,
):
    """
    Automatically annotates images using a YOLO object detection model, generating bounding box annotations in YOLO format.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        conf (float): Confidence threshold for detection model; default is 0.25.
        iou (float): IoU threshold for filtering overlapping boxes in detection results; default is 0.45.
        imgsz (int): Input image resize dimension; default is 640.
        max_det (int): Limits detections per image to control outputs in dense scenes.
        classes (list): Filters predictions to specified class IDs, returning only relevant detections.
        output_dir (str | None): Directory to save the annotated results. If None, a default directory is created.
    """
    # Load YOLO detection model
    det_model = YOLO(det_model)

    # Prepare the dataset path and output directory
    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Run YOLO detection on the dataset
    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes,
    )

    # Process detection results
    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # Get class IDs
        if len(class_ids):
            boxes = result.boxes.xyxy.tolist()  # Get bounding box coordinates

            # Save annotations to a text file
            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(boxes)):
                    x_min, y_min, x_max, y_max = boxes[i]

                    # Get original image dimensions
                    orig_h, orig_w = result.orig_img.shape[:2]

                    # Normalize bounding box coordinates
                    x_center = ((x_min + x_max) / 2) / orig_w  # Normalize x-center
                    y_center = ((y_min + y_max) / 2) / orig_h  # Normalize y-center
                    width = (x_max - x_min) / orig_w           # Normalize width
                    height = (y_max - y_min) / orig_h          # Normalize height

                    # Ensure values are within [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    # Write the annotation in YOLO format
                    f.write(f"{class_ids[i]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# Example usage
if __name__ == "__main__":
    # Define paths and parameters
    image_folder = r"D:\Desktop\yolo\an-unlabel-dataset"  # Replace with the path to your image folder
    yolo_model_path = r"note2025-latest.pt"  # Replace with your YOLO model path

    # Run the auto_annotate_bboxes_yolo_format function
    auto_annotate_bboxes_yolo_format(
        data=r"D:\Desktop\yolo\an-unlabel-dataset",
        det_model=r"D:\Desktop\yolo\note2025-latest.pt",
        conf=0.5,  # Confidence threshold for YOLO detections
        iou=0.5,  # IoU threshold for YOLO detections
        imgsz=640,  # Input image size for YOLO
        max_det=300,  # Maximum detections per image
        output_dir="yolo_annotations"  # Output directory for annotations
    )
