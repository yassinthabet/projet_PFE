import cv2
import numpy as np
import config
from utils import EuclideanDistTracker, postProcess ,detect_matricule ,count_vehicle
import requests 
import argparse
import time
import cv2
import os
import classifier
import json
# Chemin par défaut de l'image
default_image_path = "./Test_images/2.jpeg"
class VehicleCounter:
    
    def __init__(self):
        self.tracker = EuclideanDistTracker()
        self.image = cv2.imread(default_image_path)
        self.input_size = config.INPUT_SIZE
        self.confThreshold = config.CONFIDENCE_THRESHOLD
        self.nmsThreshold = config.NMS_THRESHOLD
        self.font_color = config.FONT_COLOR
        self.font_size = config.FONT_SIZE
        self.font_thickness = config.FONT_THICKNESS
        self.middle_line_position = self.image.shape[0] // 2
        self.up_line_position = self.middle_line_position - 30
        self.down_line_position = self.middle_line_position + 30
        self.is_double_clicked = False
        self.classNames = open(config.CLASSES_FILE).read().strip().split('\n')
        self.required_class_index = config.REQUIRED_CLASS_INDEX
        modelConfiguration = config.MODEL_CONFIG
        modelWeights = config.MODEL_WEIGHTS
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classNames), 3), dtype='uint8')
        self.temp_up_list = []
        self.temp_down_list = []
        self.up_list = [0, 0, 0, 0]
        self.down_list = [0, 0, 0, 0]
        cv2.namedWindow("Output")
        cv2.setMouseCallback("Output", self.set_line_position)

    def set_line_position(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.middle_line_position = y
            self.up_line_position = self.middle_line_position - 15
            self.down_line_position = self.middle_line_position + 15
            self.is_double_clicked = True
            self.down_list = [0, 0, 0, 0]

    def process_image(self):
        blob = cv2.dnn.blobFromImage(self.image, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)

        self.net.setInput(blob)
        layersNames = self.net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(outputNames)
        
        detected_vehicles = postProcess(outputs, self.image, self.colors, self.classNames, self.confThreshold, self.nmsThreshold,
                                        self.required_class_index, self.tracker, self.up_list, self.down_list, self.up_line_position,
                                        self.middle_line_position, self.down_line_position, self.temp_up_list, self.temp_down_list,
                                        self.is_double_clicked, self.font_color, self.font_size, self.font_thickness)

        
        if detected_vehicles is not None:
           for vehicle in detected_vehicles:
               x, y, w, h = vehicle['position']
               vehicle_region = self.image[y:y+h, x:x+w]
               m= detect_matricule(vehicle_region)
               print(f"Vehicle: {{'Type': '{vehicle['name']}', 'colors': '{vehicle['colors']}', 'Matricule': '{m}', 'Model': '{detection_result}'}}")


        if self.is_double_clicked:
            ih, iw, channels = self.image.shape
            cv2.line(self.image, (0, self.up_line_position), (iw, self.up_line_position), (0, 0, 255), 1)
            cv2.line(self.image, (0, self.middle_line_position), (iw, self.middle_line_position), (255, 255, 255), 2)
            cv2.line(self.image, (0, self.down_line_position), (iw, self.down_line_position), (0, 0, 255), 1)

            cv2.putText(self.image, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.image, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.image, "Car:        " + str(self.up_list[0]) + "     " + str(self.down_list[0]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.image, "Motorbike:  " + str(self.up_list[1]) + "     " + str(self.down_list[1]), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.image, "Bus:        " + str(self.up_list[2]) + "     " + str(self.down_list[2]), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.image, "Truck:      " + str(self.up_list[3]) + "     " + str(self.down_list[3]), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Output', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        

#car_make_model_classifier

# Constructeur d'arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=default_image_path,
    help="chemin vers l'image (par défaut: {})".format(default_image_path))
ap.add_argument("-y", "--yolo", default='yolo-coco',
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Chemin vers l'image
image_path = args["image"]

car_color_classifier = classifier.Classifier()

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# Charger l'image
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
outputs = net.forward(output_layers)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in outputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])

# initialize list to store detection results as JSON objects
detections = []

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        if classIDs[i] == 2:
            start = time.time()
            result = car_color_classifier.predict(image[max(y,0):y + h, max(x,0):x + w])
            end = time.time()
            # show timing information on MobileNet classifier
            print("[INFO] classifier took {:.6f} seconds".format(end - start))
            text = "{}: {:.4f}".format(result[0]['make'], float(result[0]['prob']))
            cv2.putText(image, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            cv2.putText(image, result[0]['model'], (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # store detection result as JSON object
            detection_result = {
                "make": result[0]['make'],
                "model": result[0]['model']
            }
            detections.append(detection_result)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)

    

if __name__ == "__main__":
    vc = VehicleCounter()
    vc.process_image()


