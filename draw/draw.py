import cv2
import numpy as np


class Draw():
    def __init__(self) -> None:
        pass

    def draw_annotations(self, frame, detections):
        # Split frame into channels
        blue, green, red = cv2.split(frame)

        # Create mask from one of the channels, set every pixel to true
        mask = np.ones_like(blue)
        mask = mask.astype(bool)

        # Texts to display
        texts = []

        # Loop through results
        for detection in detections:
            # Extract coordinates
            x1, y1, x2, y2 = detection['xyxy']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get center of face
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get max dimension of face
            max_dim = max(x2 - x1, y2 - y1)
            radius = int(max_dim * 0.7)

            # Create overlay, set every pixel to false
            overlay = np.zeros_like(blue)

            # Draw circle on overlay
            cv2.circle(overlay, (center_x, center_y),
                       radius, (255, 255, 255), -1)

            # Convert overlay to boolean, the circle becomes true
            overlay = overlay.astype(bool)

            # Use the circle (true) to set the masked area to false, the rest of the mask remains true.
            # This setup is applicable to multiple faces in the frame.
            mask[overlay] = False

            # Text position
            text_x = center_x + radius - 10
            text_y = center_y - radius + 10

            # Add text to texts
            texts.append(
                {'text': f"{detection['class']} {detection['track_id']}", 'xy': [text_x, text_y]})

        # Apply mask
        blue[mask] = blue[mask] * 0.5
        green[mask] = green[mask] * 0.3
        red[mask] = red[mask] * 0.2

        # Merge frame
        frame = cv2.merge((blue, green, red))

        # Draw texts
        for text in texts:
            x, y = text['xy']
            cv2.putText(frame, text['text'], (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (112, 181, 207), 2)

        return frame
