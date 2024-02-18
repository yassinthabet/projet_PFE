import math
import os
import cv2
import numpy as np
import requests
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


def get_color_name(hsv_value):
    h, s, v = hsv_value

    if v < 50:
        return "Noir"
    if s < 50:
        return "Blanc"

    color_ranges = [
        (0, 15, "Rouge"),
        (15, 45, "Jaune"),
        (45, 90, "Vert"),
        (90, 120, "Cyan"),
        (120, 150, "Bleu"),
        (150, 165, "Magenta"),
        (0, 10, "Orange"),
        (10, 20, "Gris"),
    ]

    for start, end, color in color_ranges:
        if (start <= h < end) or (165 <= h <= 180 and start == 0):
            return color

    return "Inconnu"

def get_color(vehicle_region, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max):
    if vehicle_region is None or vehicle_region.size == 0:
        return None  # Renvoie None si la région de l'image est vide

    vehicle_hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)

    if vehicle_hsv is None or vehicle_hsv.size == 0:
        return None  # Renvoie None si la conversion de couleur a échoué

    mask = cv2.inRange(vehicle_hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None  # Renvoie None si les moments sont nuls (aucun contour valide)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        hsv_value = vehicle_hsv[cy, cx]
        color_name = get_color_name(hsv_value)

        return color_name

    return None


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



def postProcess(outputs, img, colors, classNames, confThreshold, nmsThreshold, required_class_index, tracker):
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
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        vehicle_region = img[y:y+h, x:x+w]
        v1_min, v2_min, v3_min = 0, 0, 0
        v1_max, v2_max, v3_max = 255, 255, 255
        vehicle_colors = get_color(vehicle_region, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max)

        detected_vehicles.append({
            "name": name,
            "confidence": int(confidence_scores[i]*100),
            "position": (x, y, w, h),
            "colors": vehicle_colors,
        })

    closest_vehicle = find_closest_vehicle(detected_vehicles, img.shape[1])  # Pass image width for distance calculation
    return closest_vehicle