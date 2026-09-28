import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import math

# load yolov8 model
model = YOLO("best.pt")
target_classes = ["fami", "th", "vina"]

# open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# create a single kalman filter 
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])
kf.P *= 1000.0
kf.R = np.eye(2) * 5.0
kf.Q = np.eye(4)
initialized = False

# store errors
errors = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    found = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not initialized:
                    kf.x[:2] = np.array([[cx], [cy]])
                    initialized = True

                kf.predict()
                kf.update([cx, cy])
                fx, fy = int(kf.x[0]), int(kf.x[1])

                # calculate error and store
                error = math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
                errors.append(error)

                # draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)  # red = raw
                cv2.circle(frame, (fx, fy), 5, (255,0,0), -1)  # blue = filtered
                cv2.putText(frame, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                found = True
                break
        if found:
            break

    # calculate avg error and RMSE
    if errors:
        avg_err = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        cv2.putText(frame, f"Avg Error: {avg_err:.2f}px  RMSE: {rmse:.2f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Kalman Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
