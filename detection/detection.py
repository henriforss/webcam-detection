from ultralytics import YOLO
import supervision as sv


class Detection():
    def __init__(self) -> None:
        self.model = YOLO('models/best.pt')
        self.tracker = sv.ByteTrack()

    def detect_objects(self, frame):
        # Detect objects
        results = self.model.predict(frame)

        # Class names
        class_names = results[0].names

        # Convert to Supervision format
        detection_supervision = sv.Detections.from_ultralytics(results[0])

        # Update tracker with detections
        detection_with_tracks = self.tracker.update_with_detections(
            detection_supervision)

        # Extract faces
        faces = []
        for detection in detection_with_tracks:
            xyxy = detection[0].tolist()
            class_id = detection[3]
            track_id = detection[4]
            class_name = class_names[class_id]

            if class_name == 'face':
                faces.append(
                    {'xyxy': xyxy, 'class': 'face', 'track_id': track_id})

        return faces
