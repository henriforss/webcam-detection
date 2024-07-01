import cv2
from video import Video
from detection import Detection
from draw import Draw

# Initialize video capture
video = Video()

# Initialize detection
detection = Detection()

# Initialize draw
draw = Draw()

# Start video capture
cap = video.capture_video()

while True:
    # Capture frame
    _, frame = cap.read()

    # Detect objects
    detections = detection.detect_objects(frame)

    # Draw annotations
    frame = draw.draw_annotations(frame, detections)

    # Display frame
    cv2.imshow('Webcam', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
