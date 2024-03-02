import json
import time
import cv2
import numpy as np
import config
from utils import EuclideanDistTracker, postProcess
import requests 
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import torchvision
import paho.mqtt.client as mqtt
from torchvision.models import resnet50
import classifier
import cv2 as cv
import pytesseract
import re
import os
import math
import sys
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def classify_closest_vehicle(frame, net, layer_names, output_layers, colors, car_color_classifier, labels, confidence_threshold, threshold):
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward([layer_names[i - 1] for i in output_layers])
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    result = []  # Initialize result as an empty list

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, threshold)

    min_distance = float('inf')
    closest_vehicle_idx = None

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            distance = calculate_distance(W/2, H/2, x + w/2, y + h/2)

            if distance < min_distance:
                min_distance = distance
                closest_vehicle_idx = i

    if closest_vehicle_idx is not None:
        (x, y) = (boxes[closest_vehicle_idx][0], boxes[closest_vehicle_idx][1])
        (w, h) = (boxes[closest_vehicle_idx][2], boxes[closest_vehicle_idx][3])

        # draw a bounding box rectangle and label on the frame
        color = [int(c) for c in colors[classIDs[closest_vehicle_idx]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[classIDs[closest_vehicle_idx]], confidences[closest_vehicle_idx])
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        # Classification logic
        if classIDs[closest_vehicle_idx] == 2:
            result = car_color_classifier.predict(frame[max(y, 0):y + h, max(x, 0):x + w])

    (x, y) = (boxes[closest_vehicle_idx][0], boxes[closest_vehicle_idx][1])
    # assuming that 'make' is a key in the dictionary
    make_result = result[0].get('make', 'Not Found') if result else 'Not Found'
    model_result = result[0].get('model', 'Not Found') if result else 'Not Found'
    text = "{}: {:.4f}".format(make_result, float(result[0]['prob'])) if result else "No result"
    cv2.putText(frame, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    cv2.putText(frame, model_result, (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

    if not result or 'make' not in result[0] or 'model' not in result[0]:
        # Handle the case when result is an empty list or 'make'/'model' not found
        print(" Make/Model not found")

    return frame, result



# Set up YOLO
args_yolo = {
    "image": config.VIDEO_PATH,
    "yolo": 'yolo-coco',
    "confidence": 0.5,
    "threshold": 0.3
}

labelsPath = os.path.sep.join([args_yolo["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([args_yolo["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args_yolo["yolo"], "yolov3.cfg"])
net_yolo = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names_yolo = net_yolo.getLayerNames()
output_layers_yolo = net_yolo.getUnconnectedOutLayers()

# Define the car_make_model_classifier outside the loop
car_color_classifier = classifier.Classifier()



confThreshold = 0.5  
nmsThreshold = 0.4  
inpWidth = 416  
inpHeight = 416  

classesFile = "classes.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


net = cv.dnn.readNetFromDarknet(config.modelConfiguration, config.modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layerNames = net.getLayerNames()
    return [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VehicleClassifier, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
   
def preprocess_vehicle_region(vehicle_region):
    if not isinstance(vehicle_region, Image.Image):
        vehicle_region = Image.fromarray(vehicle_region)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(vehicle_region)
    image = image.unsqueeze(0)
    return image

class_names = {0: "France", 1: "Espagne"}
model = VehicleClassifier(num_classes=2)
model.load_state_dict(torch.load("Nationality.pth"))
model.eval()


def matricule(frame, outs, width_factor=1.1, height_factor=1.0):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    detected_plates = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2 * width_factor)
                top = int(center_y - height / 2 * height_factor)
                width = int(width * width_factor)
                height = int(height * height_factor)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                detected_plates.append(frame[top:top+height, left:left+width])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    detected_plates = []

    for i in range(len(indices)):
        idx = indices[i][0] if isinstance(indices[i], list) else indices[i]
        box = boxes[idx]
        left, top, width, height = box[0], box[1], box[2], box[3]
        detected_plates.append(frame[top:top+height, left:left+width])

    for plate in detected_plates:
        with torch.no_grad():
            input = preprocess_vehicle_region(plate)
            model.eval()
            predictions = model(input)
            predicted_class = torch.argmax(predictions).item()
            c = class_names.get(predicted_class, "Inconnu")
   
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(plate, config=custom_config)
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text)
       
        if cleaned_text:
            return cleaned_text, c

    return "Non detecte", "Inconnu"


class VehicleCounter:
    def __init__(self, video_path):
        self.broker_address = "127.0.0.1"
        self.broker_port = 1883
        self.topic = "vehicle_data"
        self.mqtt_client = mqtt.Client()
        self.tracker = EuclideanDistTracker()
        self.cam = cv2.VideoCapture(video_path)
        self.input_size = config.INPUT_SIZE
        self.confThreshold = config.CONFIDENCE_THRESHOLD
        self.nmsThreshold = config.NMS_THRESHOLD
        self.classNames = open(config.CLASSES_FILE).read().strip().split('\n')
        self.required_class_index = config.REQUIRED_CLASS_INDEX
        modelConfiguration = config.MODEL_CONFIG
        modelWeights = config.MODEL_WEIGHTS
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classNames), 3), dtype='uint8')
        self.last_detection_time = time.time()
        self.frame_pos = 0

    def publish_json_to_mqtt(self, json_data):
        self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
        self.mqtt_client.publish(self.topic, json_data, qos=0)
        self.mqtt_client.disconnect()
       
    def process_video(self):
        while True:
            current_time = time.time()
            elapsed_time = current_time - self.last_detection_time
        
            if elapsed_time >= 60:
                self.cam.set(cv2.CAP_PROP_POS_MSEC, self.frame_pos)  
                ret, frame = self.cam.read()
                if ret:
                   blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)
                   self.net.setInput(blob)
                   layersNames = self.net.getLayerNames()
                   outputNames = [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
                   outputs = self.net.forward(outputNames)
                   
                   frame, result = classify_closest_vehicle(frame, net_yolo, layer_names_yolo, output_layers_yolo, COLORS, car_color_classifier, LABELS,
                                            args_yolo["confidence"], args_yolo["threshold"])
                   closest_vehicle = postProcess(outputs, frame, self.colors, self.classNames, self.confThreshold, self.nmsThreshold,
                                              self.required_class_index, self.tracker)               
                   if closest_vehicle:
                    
                      cv2.imwrite("screenshot.jpg", frame)  
                      with torch.no_grad():
                        blob1 = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                        net.setInput(blob1)
                        outs1 = net.forward(getOutputsNames(net))
                        m,c=matricule(frame, outs1)
                        
                        if  c != "Inconnu" or m != "Non detecte"  :
                            make_result = result[0].get('make', 'Not Found') if result else 'Not Found'
                            model_result = result[0].get('model', 'Not Found') if result else 'Not Found'
                            json_data = {
                            "activity": "Monitoring",
                            "class": closest_vehicle['name'], 
                            "classificators": [{
                             "make": make_result,
                              "model": model_result,
                              "class": closest_vehicle['name'],
                              "color": closest_vehicle['colors'],  
                              "country": c,  
                              "registration": m
                        }],
                            "registration":m
                        }
                            json_output = json.dumps(json_data, indent=4)
                            print(json_output)
                            self.publish_json_to_mqtt(json_output)
                      self.last_detection_time = current_time
                      self.frame_pos += 1000  
                else:
                    print("finish time !!")
                    break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    vc = VehicleCounter(video_path)
    vc.process_video()