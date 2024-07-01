import cv2
from video import Video

video = Video()
cap = video.capture_video()

frame_num = 1
frames = []

while frame_num <= 100:
    _, frame = cap.read()
    print(f'Frame {frame_num} captured')
    frames.append(frame)
    frame_num += 1


cap.release()
cv2.destroyAllWindows()

# Save frames as images
for i, frame in enumerate(frames):
    cv2.imwrite(f'output/frames/frame_{i + 1}.jpg', frame)

# # Save video
# out = cv2.VideoWriter(
#     'output/video_capture.mp4', cv2.VideoWriter_fourcc(*'avc1'), 25, (640, 480))

# for frame in frames:
#     out.write(frame)
