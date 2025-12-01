import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import datetime
import os
from ultralytics import YOLO  # YOLOv8

class OptimizedHandMonitor:
    def __init__(self):
        # ---------------- AOI Polygon ----------------
        self.AOI_POLYGON = [(200, 100), (600, 100), (600, 400), (200, 400)]
        self.drawing_mode = False
        self.temp_polygon = []

        # Frame processing optimizations
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_count = 0
        self.frame_skip = 2

        # Zone tracking
        self.current_zone = None
        self.last_zone_message_time = 0
        self.zone_message_cooldown = 1.0

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # YOLO model
        # ---------- CHANGE THIS TO YOUR YOLO MODEL PATH ----------
        self.yolo_model = YOLO(r"C:\Users\ADMIN\Downloads\best.pt")

        # Pre-compile AOI zones
        self.update_compiled_polygon()

        # Video recording state
        self.recording = False
        self.out = None

        # ---------- CHANGE THIS TO YOUR SAVE FOLDER ----------
        self.save_folder = r"C:\hand detection bstone\videos Data"
        os.makedirs(self.save_folder, exist_ok=True)

    def update_compiled_polygon(self):
        """Pre-compile polygon and create top/bottom zones (yellow/red)"""
        self.compiled_polygon = np.array(self.AOI_POLYGON, dtype=np.int32)
        x_coords = [p[0] for p in self.AOI_POLYGON]
        y_coords = [p[1] for p in self.AOI_POLYGON]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        bbox_height = max_y - min_y
        split_y = min_y + int(bbox_height * 0.1)  # top 60% yellow, bottom 40% red

        self.yellow_zone_polygon = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, split_y),
            (min_x, split_y)
        ]
        self.red_zone_polygon = [
            (min_x, split_y),
            (max_x, split_y),
            (max_x, max_y),
            (min_x, max_y)
        ]

        self.compiled_yellow_zone = np.array(self.yellow_zone_polygon, dtype=np.int32)
        self.compiled_red_zone = np.array(self.red_zone_polygon, dtype=np.int32)

    def draw_polygon(self, event, x, y, flags, param):
        if self.drawing_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_polygon.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.temp_polygon) >= 3:
                    self.AOI_POLYGON = self.temp_polygon.copy()
                    self.update_compiled_polygon()
                    print(f"[INFO] AOI Polygon updated: {self.AOI_POLYGON}")
                    self.temp_polygon = []
                    self.drawing_mode = False
                else:
                    print("[WARNING] At least 3 points are required.")

    def point_in_poly_fast(self, pt, polygon):
        return cv2.pointPolygonTest(polygon, pt, False) >= 0

    def get_hand_zone(self, pt):
        if self.point_in_poly_fast(pt, self.compiled_red_zone):
            return 'red'
        elif self.point_in_poly_fast(pt, self.compiled_yellow_zone):
            return 'yellow'
        else:
            return None

    def frame_capture_thread(self, cap):
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass

    def process_hands_optimized(self, img):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hand_detected = False
        current_zone = None

        # ---------- MediaPipe detection ----------
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in lm.landmark]
                zone_detected = None
                for lx, ly in landmarks:
                    zone = self.get_hand_zone((lx, ly))
                    if zone == "red":
                        zone_detected = "red"
                        break
                    elif zone == "yellow" and zone_detected != "red":
                        zone_detected = "yellow"
                    cv2.circle(img, (lx, ly), 4, (255, 255, 255), -1)

                if zone_detected:
                    hand_detected = True
                    current_zone = zone_detected
                    color = (0, 0, 255) if zone_detected == "red" else (0, 255, 255)
                    current_time = time.time()
                    if current_time - self.last_zone_message_time > self.zone_message_cooldown:
                        if zone_detected != self.current_zone:
                            if zone_detected == 'yellow':
                                print("[WARNING] MACHINE SLOWING DOWN - Finger in Yellow Zone [MediaPipe]")
                            elif zone_detected == 'red':
                                print("[EMERGENCY] EMERGENCY STOP - Finger in Red Zone [MediaPipe]")
                            self.current_zone = zone_detected
                            self.last_zone_message_time = current_time
                else:
                    color = (0, 255, 0)
        # ---------- YOLO fallback ----------
        else:
            yolo_results = self.yolo_model(img, verbose=False)
            for r in yolo_results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), conf in zip(boxes, confs):
                    if conf < 0.5:
                        continue

                    points_to_check = [
                        (int(x1), int(y1)),
                        (int(x2), int(y1)),
                        (int(x2), int(y2)),
                        (int(x1), int(y2)),
                        (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    ]

                    zone_detected = None
                    for pt in points_to_check:
                        zone = self.get_hand_zone(pt)
                        if zone == "red":
                            zone_detected = "red"
                            break
                        elif zone == "yellow" and zone_detected != "red":
                            zone_detected = "yellow"

                    if zone_detected:
                        hand_detected = True
                        current_zone = zone_detected
                        color = (0, 0, 255) if zone_detected == "red" else (0, 255, 255)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cx_box = int((x1 + x2) / 2)
                        cy_box = int((y1 + y2) / 2)
                        cv2.circle(img, (cx_box, cy_box), 5, color, -1)
                        current_time = time.time()
                        if current_time - self.last_zone_message_time > self.zone_message_cooldown:
                            if zone_detected != self.current_zone:
                                if zone_detected == 'yellow':
                                    print("[WARNING] MACHINE SLOWING DOWN - Hand in Yellow Zone [YOLO]")
                                elif zone_detected == 'red':
                                    print("[EMERGENCY] EMERGENCY STOP - Hand in Red Zone [YOLO]")
                                self.current_zone = zone_detected
                                self.last_zone_message_time = current_time

        if not hand_detected and self.current_zone is not None:
            print("[INFO] Hand left safety zone")
            self.current_zone = None

        return hand_detected, current_zone

    def draw_ui_optimized(self, img, hand_detected, current_zone):
        h, w = img.shape[:2]

        if not self.drawing_mode:
            overlay = img.copy()
            yellow_poly = np.array(self.yellow_zone_polygon, np.int32)
            cv2.fillPoly(overlay, [yellow_poly], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [yellow_poly], True, (0, 255, 255), 2)

            overlay = img.copy()
            red_poly = np.array(self.red_zone_polygon, np.int32)
            cv2.fillPoly(overlay, [red_poly], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [red_poly], True, (0, 0, 255), 2)

        display_poly = self.temp_polygon if self.drawing_mode else self.AOI_POLYGON
        if len(display_poly) >= 2:
            poly_array = np.array(display_poly, np.int32)
            cv2.polylines(img, [poly_array], not self.drawing_mode, (255, 255, 255), 3)

        if current_zone == 'red':
            split_y = self.red_zone_polygon[0][1]
            text = "STOP"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = split_y + (h - split_y + text_size[1]) // 2
            cv2.putText(img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        elif current_zone == 'yellow':
            split_y = self.yellow_zone_polygon[2][1]
            text = "WARNING"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (split_y // 2) + text_size[1] // 2
            cv2.putText(img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
        elif hand_detected:
            cv2.putText(img, "HAND DETECTED - SAFE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(img, "INVICTUS SOLUTION", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 3)

    def run(self):
        ip_url = "rtsp://admin:Techno%40123@192.168.1.64:554/Streaming/Channels/101"
        cap = cv2.VideoCapture(ip_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[ERROR] Failed to open RTSP stream")
            return

        capture_thread = threading.Thread(target=self.frame_capture_thread, args=(cap,), daemon=True)
        capture_thread.start()

        cv2.namedWindow("Hand AOI Monitor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Hand AOI Monitor", self.draw_polygon)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v' for mp4

        while True:
            try:
                img = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.frame_count += 1

            if self.frame_count % self.frame_skip != 0:
                self.draw_ui_optimized(img, False, None)
                cv2.imshow("Hand AOI Monitor", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            hand_detected, current_zone = self.process_hands_optimized(img)
            self.draw_ui_optimized(img, hand_detected, current_zone)

            # ---------- RECORDING LOGIC ----------
            if current_zone == "red":
                if not self.recording:
                    filename = datetime.datetime.now().strftime("intrusion_%Y%m%d_%H%M%S.avi")
                    filepath = os.path.join(self.save_folder, filename)
                    self.out = cv2.VideoWriter(filepath, fourcc, 15.0, (img.shape[1], img.shape[0]))
                    self.recording = True
                    print(f"[RECORDING STARTED] {filepath}")
                if self.out is not None:
                    self.out.write(img)
            else:
                if self.recording:
                    self.recording = False
                    if self.out is not None:
                        self.out.release()
                        self.out = None
                    print("[RECORDING STOPPED]")

            cv2.imshow("Hand AOI Monitor", img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('d'):
                self.drawing_mode = True
                self.temp_polygon = []
            elif key == ord('r'):
                self.AOI_POLYGON = [(200, 100), (600, 100), (600, 400), (200, 400)]
                self.update_compiled_polygon()

        cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor = OptimizedHandMonitor()
    monitor.run()
