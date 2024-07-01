import cv2


class Video():
    def __init__(self) -> None:
        pass

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        return cap
