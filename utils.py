import math
import os
import cv2
import numpy as np
from collections import Counter


class EuclideanDistTracker:

    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def dist_calculator(box_width, img_w):
    focal_length = 1000  
    known_width = 100  
    distance = (known_width * focal_length) / box_width
    return distance

def find_closest_vehicle(detected_vehicles, img_width):
    if not detected_vehicles:
        return None

    min_distance = float('inf')
    closest_vehicle = None

    for vehicle in detected_vehicles:
        x, y, w, h = vehicle["position"]
        box_width = w
        distance = dist_calculator(box_width, img_width)
        if distance < min_distance:
            min_distance = distance
            closest_vehicle = vehicle

    return closest_vehicle



def postProcess(outputs, img,  classNames, confThreshold, nmsThreshold, required_class_index, tracker):
    detected_vehicles = []
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index and confidence > confThreshold:
                w, h = int(det[2] * width), int(det[3] * height)
                x, y = int((det[0] * width) - w/2), int((det[1] * height) - h/2)
                boxes.append([x, y, w, h])
                classIds.append(classId)
                confidence_scores.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        
        name = classNames[classIds[i]]
        
        

        detected_vehicles.append({
            "name": name,
            "confidence": int(confidence_scores[i]*100),
            "position": (x, y, w, h),
            
        })

    closest_vehicle = find_closest_vehicle(detected_vehicles, img.shape[1])  # Pass image width for distance calculation
    return closest_vehicle