import cv2
import torch
import os
from ultralytics import YOLO

MODEL_PATH = "/home/mahanth/MTP/models/yolov8n_best.pt"
VIDEO_PATH = "data/input/night_view.mp4"
OUTPUT_PATH = "data/output/night_view_yolov8n_best_1.mp4"

CONF_THRES = 0.45  
IOU_THRES = 0.25    # add this
FRAME_SIZE = (640, 360)


model = YOLO(MODEL_PATH)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Error opening video: {VIDEO_PATH}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = fps if fps > 0 else 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, FRAME_SIZE)

# -----------------------------
# Warm-up (important for stability)
# -----------------------------
dummy = torch.zeros(1, 3, 640, 640).to(device)
model(dummy)

# -----------------------------
# Main loop
# -----------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize WITHOUT distortion (letterbox style)
    frame_resized = cv2.resize(frame, FRAME_SIZE)

    # -----------------------------
    # Tracking
    # -----------------------------
    results = model.track(
        frame_resized,
        conf=CONF_THRES,
        iou=IOU_THRES,              # 🔥 reduces duplicate boxes
        persist=True,
        device=device,
        tracker="bytetrack.yaml",
        verbose=False,
        agnostic_nms=False          # 🔥 VERY IMPORTANT FIX
    )

    # -----------------------------
    # Plot
    # -----------------------------
    if results[0].boxes is not None:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame_resized

    out.write(annotated_frame)

    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done. Saved to: {OUTPUT_PATH}")