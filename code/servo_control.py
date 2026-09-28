import cv2
from ultralytics import YOLO
import pyfirmata
from time import sleep
from itertools import combinations

# Setup YOLO + Servo
model = YOLO("best.pt")
target_classes = ["th", "vina", "fami"]

board = pyfirmata.Arduino('COM3')  
servo_x = board.get_pin('d:3:s')
servo_y = board.get_pin('d:2:s')
sleep(2)

cap = cv2.VideoCapture(0)
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# X_SERVO_MIN, X_SERVO_MAX = 45, 135
# Y_SERVO_MIN, Y_SERVO_MAX = 60, 120
# X_SERVO_MIN = 65   # more higher less right
# X_SERVO_MAX = 120 # more lower less left
# Y_SERVO_MIN = 60
# Y_SERVO_MAX = 115 # more lower more higher
X_SERVO_MIN = 65   # more higher less right
X_SERVO_MAX = 115 # more lower less left
Y_SERVO_MIN = 60
Y_SERVO_MAX = 70 # more lower more higher

def map_position_to_angle(cx, cy):
    cx = FRAME_WIDTH - cx  # Flip camera X
    angle_x = int(X_SERVO_MIN + (cx / FRAME_WIDTH) * (X_SERVO_MAX - X_SERVO_MIN))
    angle_y = int(Y_SERVO_MIN + (cy / FRAME_HEIGHT) * (Y_SERVO_MAX - Y_SERVO_MIN))
    return max(X_SERVO_MIN, min(angle_x, X_SERVO_MAX)), max(Y_SERVO_MIN, min(angle_y, Y_SERVO_MAX))

# MST with Kruskal 

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, u):
        while u != self.parent[u]:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        self.parent[pu] = pv
        return True

def kruskal_mst(points):
    n = len(points)
    edges = []
    for (i, p1), (j, p2) in combinations(enumerate(points), 2):
        dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        edges.append((dist, i, j))
    edges.sort()

    uf = UnionFind(n)
    mst = [[] for _ in range(n)]
    for dist, u, v in edges:
        if uf.union(u, v):
            mst[u].append(v)
            mst[v].append(u)
    return mst

def dfs_traverse(mst, start=0):
    visited = set()
    order = []

    def dfs(u):
        visited.add(u)
        order.append(u)
        for v in mst[u]:
            if v not in visited:
                dfs(v)
    dfs(start)
    return order

#  Main Loop 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    targets = []  # list of (cx, cy) of valid objects

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                targets.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if len(targets) >= 2:
        mst = kruskal_mst(targets)
        order = dfs_traverse(mst, start=0)  

        for idx in order:
            cx, cy = targets[idx]
            angle_x, angle_y = map_position_to_angle(cx, cy)
            servo_x.write(angle_x)
            sleep(0.015)
            servo_y.write(angle_y)
            sleep(0.015)
            print(f"Pointing to {cx},{cy} → angle ({angle_x}, {angle_y})")
            sleep(3)  # Wait on each target

    cv2.imshow("YOLO + MST Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
