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

# Chargement du modèle de détection de couleur
final_model = torch.load('./final_model_85.t', map_location='cpu')

# Définition des transformations pour l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class_name = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

# Fonction pour obtenir les coordonnées de la boîte englobante du véhicule
def get_box(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    
    # Charger le modèle de détection d'objets (exemple : modèle YOLO)
    net = cv2.dnn.readNetFromDarknet("./yolo-coco/yolov3.cfg", "./yolo-coco/yolov3.weights")  # Utiliser le chemin approprié pour votre modèle
    
    # Obtenir les noms des couches de sortie
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Redimensionner l'image et la passer à travers le réseau
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Analyser les sorties du réseau
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Identifier la classe pour les véhicules (ex: car)
                # Récupérer les coordonnées de la boîte englobante
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Supprimer les détections non précises en utilisant la suppression non maximale
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Renvoyer les coordonnées de la boîte englobante du premier véhicule détecté
    if len(indices) > 0:  # Vérifier si des indices sont retournés
        i = indices.flatten()  # Aplatir les indices
        return boxes[i[0]]  # Utiliser le premier indice retourné
    else:
        return None


# Fonction pour recadrer l'image du véhicule
def crop_car(src_path, x, y, w, h):
    src_image = cv2.imread(src_path)
    if src_image is None:
        return
    crop_image = src_image[y:y+h, x:x+w]
    dst_img = cv2.resize(src=crop_image, dsize=(224, 224))
    img = Image.fromarray(dst_img)
    image = transform(img).float()
    image = torch.Tensor(image)
    return image.unsqueeze(0)

# Fonction pour prédire la couleur du véhicule
def predict_color(src):
    box = get_box(src)
    if box is None:
        return "error"
    image = crop_car(src, *box)
    preds = final_model(image)
    return class_name[int(preds.max(1)[1][0])]


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
    def __init__(self):
        self.broker_address = "127.0.0.1"
        self.broker_port = 1883
        self.topic = "vehicle_data"
        self.mqtt_client = mqtt.Client()
        self.tracker = EuclideanDistTracker()
        self.cam = cv2.VideoCapture(config.VIDEO_PATH)
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
        self.mqtt_client.publish(self.topic, json_data)
        self.mqtt_client.disconnect()
        
    def process_video(self):
        while True:
            current_time = time.time()
            elapsed_time = current_time - self.last_detection_time
        
            if elapsed_time >= 60:
                self.cam.set(cv2.CAP_PROP_POS_MSEC, self.frame_pos)  
                ret, frame = self.cam.read()
                if ret:
                   t = self.cam.get(cv2.CAP_PROP_POS_MSEC)
                   print(f"Temps écoulé: {t:.2f} ms")
                   print(f"Frame {self.frame_pos} lue")
                   blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)
                   self.net.setInput(blob)
                   layersNames = self.net.getLayerNames()
                   outputNames = [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
                   outputs = self.net.forward(outputNames)
                   
                   closest_vehicle = postProcess(outputs, frame, self.classNames, self.confThreshold, self.nmsThreshold,
                                              self.required_class_index, self.tracker)               
                   if closest_vehicle:
                    
                      cv2.imwrite("screenshot.jpg", frame)  
                      with torch.no_grad():
                        blob1 = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                        net.setInput(blob1)
                        outs1 = net.forward(getOutputsNames(net))
                        m,c=matricule(frame, outs1)
                        
                        if  c != "Inconnu" or m != "Non detecte"  :
                            json_data = {
                            "activity": "Monitoring",
                            "class": closest_vehicle['name'], 
                            "classificators": [{
                              "class": closest_vehicle['name'],
                              "color": predict_color("screenshot.jpg"),
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
    vc = VehicleCounter()
    vc.process_video()