import numpy as np
import cv2
import torch
import time
import sqlite3
import base64
import logging
logging.basicConfig(
    filename="events_log.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

class EventDetector():
    def __init__(self):
        self.MIN_CONF_THRESHOLD = 0.3
        self.PERSON_CLASS_ID = 0
        self.DB_DISK_PATH = "events.db"
        self.MIN_EVENT_GAP_SECONDS = 60
        
        self.logger = logging.getLogger(__name__)
        
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n")
        #self.dbConnection = sqlite3.connect(self.DB_DISK_PATH)
        #self.dbCursor = self.dbConnection.cursor()
        with sqlite3.connect(self.DB_DISK_PATH) as conn:
            conn.execute("""
                            CREATE TABLE IF NOT EXISTS events (
                                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                                image_file_path      TEXT    NOT NULL,
                                x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
                                created_at TEXT DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
        
    def cropImage(self, frame, bbox):
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    def detect(self, frame):
        results = self.model(frame)
        #results.print()
        #results.show()
        return results # models.common.detections
    
    def processDetections(self, results):
        person_detected = False
        person_bbox = [-1, -1, -1, -1]
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cls = int(cls)
            if cls == self.PERSON_CLASS_ID and conf > self.MIN_CONF_THRESHOLD:
                person_detected = True
                person_bbox = [x1, y1, x2, y2]
                break
            self.logger.info("%s %.2f", cls, conf)
            
        return person_detected, person_bbox
    
    def drawDetection(self, frame, bbox):
        cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), thickness=3)
    
    def img_to_str(self, frame):
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("encode failed")
        return base64.b64encode(buf).decode("ascii")

    def str_to_img(self, s):
        buf = base64.b64decode(s)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    def saveInDB(self, bbox, image_file_path):
        
        with sqlite3.connect(self.DB_DISK_PATH) as conn:
            conn.execute("INSERT INTO events (image_file_path, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?)", (image_file_path, bbox[0], bbox[1], bbox[2], bbox[3]))
    
    def displayDB(self):
        with sqlite3.connect(self.DB_DISK_PATH) as conn:
            for row in conn.execute("""
                                        SELECT * FROM (
                                            SELECT id, image_file_path, x1, y1, x2, y2, created_at
                                            FROM events
                                            ORDER BY id DESC
                                            LIMIT 3
                                        ) ORDER BY id ASC
                                    """):
                self.logger.info("[%s] -\"%s\"- %s,%s,%s,%s %s", row[0], row[1], row[2], row[3], row[4], row[5], row[6])
        
    def saveEvent(self, frame, bbox, timestamp, crop=True):
        saveImage = frame
        
        timestamp_str = time.strftime("%Y_%m_%d_%H_%M_%S",  time.localtime(timestamp))
        image_file_path = f"person_{timestamp_str}.jpg"
        self.saveInDB(bbox, image_file_path)
        
        if crop:
            saveImage = self.cropImage(frame, bbox)

        cv2.imwrite(image_file_path, saveImage)
        self.drawDetection(frame, bbox)
        cv2.imshow("", saveImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        lastEventTimestamp = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detect(frame)
            person_detected, person_bbox = self.processDetections(results)
            
            if person_detected:
                currentEventTimestamp = time.time()
                if lastEventTimestamp is None or (currentEventTimestamp - lastEventTimestamp  > self.MIN_EVENT_GAP_SECONDS):
                    lastEventTimestamp = time.time()
                    
                    self.saveEvent(frame, person_bbox, lastEventTimestamp, crop=False)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.displayDB()
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    det = EventDetector() # simple single person detector
    det.run()
    
if __name__ == "__main__":
    main()