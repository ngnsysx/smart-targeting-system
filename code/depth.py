import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Open stereo cameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)
cap_left.set(3, 640)
cap_left.set(4, 480)
cap_right.set(3, 640)
cap_right.set(4, 480)

# Stereo matcher setup
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=112,
    blockSize=15,
    P1=8*3*15**2,
    P2=32*3*15**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# WLS filter
wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls.setLambda(8000)
wls.setSigmaColor(1.5)
right_matcher = cv2.ximgproc.createRightMatcher(stereo)

# Calibration parameters
focal_length = 1360.0
baseline = 0.09

# ROI and smoothing
roi_percent = 0.33
last_distances = []
last_red_percentages = []

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not retL or not retR:
        break

    grayL = cv2.equalizeHist(cv2.GaussianBlur(cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    grayR = cv2.equalizeHist(cv2.GaussianBlur(cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY), (5, 5), 0))

    dispL = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    dispR = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0
    disp = wls.filter(dispL, grayL, None, dispR)

    norm_disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)

    h, w = disp.shape
    roi_x, roi_y = int(w * roi_percent), int(h * roi_percent)
    roi_w, roi_h = int(w * roi_percent), int(h * roi_percent)

    roi_disp = disp[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    valid = roi_disp[(roi_disp > 1.0) & (roi_disp < 112.0)]
    if len(valid) > 0:
        top_disp = np.sort(valid)[-10:]
        raw_distance = (focal_length * baseline) / np.mean(top_disp)
    else:
        raw_distance = 3.0

    roi_color = depth_map[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    b, g, r = cv2.split(roi_color)
    red_mask = np.logical_and(r > 180, np.logical_and(g < 120, b < 100))
    red_visual = np.zeros_like(roi_color)
    red_visual[red_mask] = [0, 0, 255]

    red_percentage = np.sum(red_mask) / (roi_w * roi_h)
    last_red_percentages.append(red_percentage)
    if len(last_red_percentages) > 5:
        last_red_percentages.pop(0)
    stable_red_percentage = np.mean(last_red_percentages)
    red_punishment = stable_red_percentage * 9
    corrected_distance = max(0.1, min(raw_distance - red_punishment, 3.0))

    last_distances.append(corrected_distance)
    if len(last_distances) > 5:
        last_distances.pop(0)
    smoothed_distance = np.mean(last_distances)

    # YOLO detection
    results = model(frameL, stream=True)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frameL, f"{label} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Annotate ROI and distance info
    cv2.rectangle(frameL, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
    cv2.rectangle(depth_map, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)

    cv2.putText(frameL, f"Raw: {raw_distance:.2f}m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frameL, f"Red%: {stable_red_percentage:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frameL, f"Punish: {red_punishment:.2f}m", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display windows
    cv2.imshow("YOLO + Depth", frameL)
    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("Red Detection ROI", red_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
