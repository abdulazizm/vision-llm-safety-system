# pip install opencv-python torch torchvision ultralytics pandas tqdm seaborn scipy
# curl -O https://raw.githubusercontent.com/ultralytics/yolov5/refs/heads/master/requirements.txt
# pip install -r requirements.txt

#TODO:
#replace prints with logging
#log in db - sqlite

import numpy as np
import cv2
import torch
import time

class EventDetector():
    def __init__(self):
        self.MIN_CONF_THRESHOLD = 0.3
        self.PERSON_CLASS_ID = 0
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n")
        
    def cropImage(self, frame, bbox):
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    def detect(self, frame):
        results = self.model(frame)
        results.print()
        #results.show()
        return results # models.common.detections
    
    def processDetections(self, results):
        person_detected = False
        person_bbox = [-1, -1, -1, -1]
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cls = int(cls)
            if cls == self.PERSON_CLASS_ID and conf > self.MIN_CONF_THRESOLD:
                person_detected = True
                person_bbox = [x1, y1, x2, y2]
                break
            print(cls, conf)
            
        return person_detected, person_bbox
    
    def drawDetection(self, frame, bbox):
        cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), thickness=3)
    
    def saveEvent(self, frame, bbox, crop=True):
        saveImage = frame
        if crop:
            saveImage = self.cropImage(frame, bbox)
        else:
            self.drawDetection(frame, bbox)
        
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S",  time.localtime(time.time()))
        cv2.imwrite(f"person_{timestamp}.jpg", saveImage)
        cv2.imshow("", saveImage)
        cv2.waitKey(1)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(np.shape(frame))
            
            results = self.detect(frame)
            person_detected, person_bbox = self.processDetections(results)
            
            if person_detected:
                self.saveEvent(frame, person_bbox, crop=False)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    det = EventDetector() # simple single person detector
    det.run()
    
if __name__ == "__main__":
    main()
