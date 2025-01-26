from ultralytics import YOLO
import cv2
import numpy as np
from math import ceil


class Detect:
    def __init__(self, model: str = "frc2025-3_ncnn_model", conf: float = 0.8, datastream: bool = False, confident_stander: float = 0.7) -> None:
        '''
        init detection
        '''
        self.conf = conf
        self.confident_stander = confident_stander  # Confidence threshold
        self.model = YOLO(model)
        self.ClassInt: list[int] = [i for i in range(len(self.model.names))]
        self.image_saver = ImageSaver()  # Initialize the ImageSaver class

    def __CamInit__(self, cam_id: int, resolution: tuple[int, int] = (480, 640)):
        '''
        init camera
        '''
        cam = cv2.VideoCapture(cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        return cam

    def static(self, imgs: list[str], save: bool = True):
        '''
        detect object from normal image
        '''
        for img in imgs:
            self.model(img, save=save, conf=self.conf)

    def stream(self, camera: int = 0, resolution: tuple[int, int] = (480, 640), cls: list[int] = None, gui: bool = True) -> None:
        '''
        detect image from webcam
        '''
        cam = self.__CamInit__(camera, resolution)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        color = (255, 255, 0)
        thickness = 2

        print("Model now running...")
        while True:
            success, img = cam.read()
            # Perform inference with streaming on the current frame
            results = self.model(img, stream=True, conf=self.conf, classes=cls)

            for res in results:
                if not res.boxes:
                    self.DeltaDistance = 3000
                    continue  # Skip if no boxes detected

                # Vectorized calculation of distances between box centers and image center
                boxes = res.boxes
                box_centers = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) / 2
                distances = np.abs(box_centers - resolution[1] / 2)
                nearest_idx = np.argmax(distances.cpu().numpy())

                # Draw rectangles around detected objects
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()  # Confidence of the current detection
                    class_id = int(box.cls)

                    
                    # Draw bounding box
                    if i == nearest_idx:
                        if (resolution[1] / 2 + 10) >= box_centers[i] >= (resolution[1] / 2 - 10):  # if nearest detected and centered
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        else:  # if nearest detected but not centered
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # others (multi-detected processing)

                    # Display detection details
                    cv2.putText(img, f"Detected {len(boxes)} item(s)", (10, 23), font, fontscale, color, thickness)
                    cv2.putText(img, f"{ceil(conf * 100)}%  {x1}x{y1}, {x2}x{y2}", (x1, y1), font, fontscale, color, thickness)
                    cv2.putText(img, f"{self.model.names[class_id]}", (x1 - 30, y1 - 30), font, fontscale, color, thickness)

            cv2.imshow("CamDetected", img) if gui else None

            # Break the loop on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()


class Nano:
    def __init__(self):
        pass


if __name__ == "__main__":
    Detect("FRC_ncnn_model", datastream=True, conf=0.7).stream(camera=0, gui=True)
