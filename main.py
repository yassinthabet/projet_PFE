import json
import time
import cv2
import numpy as np
import config
from utils import EuclideanDistTracker, postProcess
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import paho.mqtt.client as mqtt
from torchvision.models import resnet50
import cv2 as cv
import pytesseract
import re
import http.client as httplib
import sys
import traceback

confThreshold = 0.5  
nmsThreshold = 0.4  
inpWidth = 416  
inpHeight = 416  

classesFile = "classes.names"
classes = None
try:
    with open(config.CLASSES_FILE, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: Classes file '{config.CLASSES_FILE}' not found.")
    sys.exit(1)

modelConfiguration = "./matricule_model/darknet-yolov3.cfg"
modelWeights = "./matricule_model/model.weights"

try:
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
except cv2.dnn.NetBackendNotAvailable:
    print("Error: OpenCV DNN backend not available. Please ensure you have the required libraries installed.")
    sys.exit(1)
except cv2.dnn.OpenCVError as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    try:
        layerNames = net.getLayerNames()
        return [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception as e:
        print("Erreur lors de la récupération des noms de sortie:", e)

class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VehicleClassifier, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
    
def preprocess_vehicle_region(vehicle_region):
    try:
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
    except Exception as e:
        print("Erreur lors du prétraitement de la région du véhicule:", e)

class_names = {0: "France", 1: "Espagne"}

try:
    model = VehicleClassifier(num_classes=2)
    model.load_state_dict(torch.load("Nationality.pth"))
    model.eval()
except Exception as e:
    print(f"Erreur lors du chargement du modèle de classification de nationalité : {e}")
    sys.exit()


def matricule(frame, outs, width_factor=1.1, height_factor=1.0):
    try:
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

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        detected_plates = []

        for i in range(len(indices)):
            idx = indices[i][0] if isinstance(indices[i], list) else indices[i]
            box = boxes[idx]
            left, top, width, height = box[0], box[1], box[2], box[3]
            detected_plates.append(frame[top:top+height, left:left+width])

        for plate in detected_plates:
            with torch.no_grad():
                input_plate = cv2.imwrite("plate_temp.jpg", plate)
                input_plate = cv2.imread("plate_temp.jpg")
                input_tensor = preprocess_vehicle_region(input_plate)
                model.eval()
                predictions = model(input_tensor)
                predicted_class = torch.argmax(predictions).item()
                c = class_names.get(predicted_class, "Inconnu")

            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(plate, config=custom_config)
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text)

            if cleaned_text:
                return cleaned_text, c

        return "Non detecte", "Inconnu"
    except Exception as e:
        print("Erreur lors de la détection de la plaque d'immatriculation:", e)

def load_checkpoint(filepath):
    try:
                                                
       model = models.resnet34(pretrained=True)                          
       num_ftrs = model.fc.in_features                                   
       model.fc = nn.Linear(num_ftrs, 140)                               
       checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
       model.load_state_dict(checkpoint['state_dict'], strict=False)     
       model.class_to_idx = checkpoint['class_to_idx']                   
                                                                      
       return model 
    except Exception as e:
        print("Erreur lors du chargement du point de contrôle:", e)

brand = load_checkpoint('brand.pth')                                  
brand.eval() 
with open('marque.json', 'r') as f:                                   
    class_to_idx = json.load(f)                                       
                                                                      
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}


img_transforms = transforms.Compose([                                 
    transforms.Resize((244, 244)),                                    
    transforms.CenterCrop(224),                                       
    transforms.ToTensor(),                                            
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))            
])                                                                    
                                                                      
def tansf_image(image_path):
    try :                                          
       image = Image.open(image_path)                                    
       image = img_transforms(image).float()                             
       image = image.unsqueeze(0)                                        
       return image
    except Exception as e:
        print("Erreur lors du chargement de l'image:", e)
    
    
class VehicleCounter:
    def __init__(self, video_path):
        try:
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
        except Exception as e:
            print("Erreur lors de la lecture du fichier de classes:", e)

    def publish_json_to_mqtt(self, json_data):
        try:
           self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
           self.mqtt_client.publish(self.topic, json_data, qos=0)
           self.mqtt_client.disconnect()
        except Exception as e:
            print("Erreur lors de l'initialisation:", e)
            traceback.print_exc()
            
    def process_video(self):
        s = time.time()
        while self.cam.isOpened():           
            ret, frame = self.cam.read()      
            if ret:
                try:
                    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)
                    self.net.setInput(blob)
                    layersNames = self.net.getLayerNames()
                    outputNames = [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
                    outputs = self.net.forward(outputNames)
                
                    closest_vehicle = postProcess(outputs, frame, self.colors, self.classNames, self.confThreshold, self.nmsThreshold, self.required_class_index, self.tracker)               
                    if closest_vehicle:
                        try:
                            cv2.imwrite("screenshot.jpg", frame) 
                        except Exception as img_err:
                            print("Erreur lors de l'enregistrement de l'image:", img_err)
                            continue
                    
                        img = "screenshot.jpg" 
                        imgtr = tansf_image(img)
                        with torch.no_grad():
                            blob1 = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                            net.setInput(blob1)
                            outs1 = net.forward(getOutputsNames(net))
                            try:
                                m, c = matricule(frame, outs1)
                            except Exception as mat_err:
                                print("Erreur lors de la détection de la plaque d'immatriculation:", mat_err)
                                continue

                            try:
                                output = brand(imgtr)
                                probabilities = torch.exp(output)
                                dim = 1               
                                top_prob, top_class = probabilities.topk(1, dim)
                                predicted_class_index = top_class.item()        
                                predicted_class_name = idx_to_class[predicted_class_index]
                                make, model = predicted_class_name.split(' ', 1)
                            except Exception as brand_err:
                                print("Erreur lors de la détection de la marque:", brand_err)
                                continue
                        
                            try:
                                json_data = {                                   
                                    "activity": "Monitoring",                 
                                    "class": closest_vehicle['name'],         
                                    "classificators": [{                      
                                        "make": make,                           
                                        "model": model,                         
                                        "class": closest_vehicle['name'],     
                                        "color": closest_vehicle['colors'],  
                                        "country": c,                           
                                        "registration": m                       
                                    }],                                           
                                    "registration": m                          
                                }                                             
                                json_output = json.dumps(json_data, indent=4)
                                print(json_output)
                                self.publish_json_to_mqtt(json_output)
                                print("Message sent") 
                            except Exception as json_err:
                                print("Erreur lors de la génération du JSON:", json_err)
                                continue
                except Exception as err:
                    print("Une erreur est survenue lors du traitement de la vidéo:", err)
                    continue
            else:
                print("time finished !!")
                break




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    vc = VehicleCounter(video_path)
    vc.process_video()