import cv2

# These are the paths and the runtime configurations.
VIDEO_PATH = "data/lot.mp4"
OUT_PATH = "data/frame_video.jpg"
TIME_MILLISECONDS = 1.0

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
# Here, I check the video's frame rate.
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0
# The reference frame is extracted in this section.
cap.set(cv2.CAP_PROP_POS_MSEC, TIME_MILLISECONDS)
ok, frame = cap.read()
cap.release()
cv2.imwrite(OUT_PATH, frame)
print("Wrote:", OUT_PATH, "shape:", frame.shape, "fps:", fps, "time(ms):", TIME_MILLISECONDS)