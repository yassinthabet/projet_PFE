import cv2 as cv
import numpy as np
import pytesseract
import re
import time
import config
import cv2 
import os 
import classifier

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image


image_path = "./F2.jpg"
classesFile = "classes.names"

net = cv.dnn.readNetFromDarknet("./matricule_model/darknet-yolov3.cfg", "./matricule_model/model.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layerNames = net.getLayerNames()
    return [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

def matricule(frame, outs, width_factor=1.1, height_factor=1.0):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

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
                # Multiplier la largeur par un certain facteur
                width = int(width * width_factor)
                height = int(height * height_factor)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    detected_plates = []
    for i in range(len(indices)):
        idx = indices[i][0] if isinstance(indices[i], list) else indices[i]
        box = boxes[idx]
        left, top, width, height = box[0], box[1], box[2], box[3]
        detected_plates.append(frame[top:top+height, left:left+width])

    for plate in detected_plates:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(plate, config=custom_config)
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text)  # Supprimer les symboles spécifiés et les espaces
        
        return cleaned_text



frame = cv.imread(image_path)
blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
net.setInput(blob)
outs = net.forward(getOutputsNames(net))
a = matricule(frame, outs)

#car_make_model_classifier
         
# Définir les arguments directement dans le code
args = {
    "image": config.VIDEO_PATH,
    "yolo": 'yolo-coco',
    "confidence": 0.5,
    "threshold": 0.3
}

# Chemin vers l'image
image_path = args["image"]

# Car Make Model Classifier
car_color_classifier = classifier.Classifier()

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Print the paths for debugging
print("Config Path:", configPath)
print("Weights Path:", weightsPath)

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Charger l'image
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Pause entre les modèles (ajustez la durée selon vos besoins)
time.sleep(5)  # 5 secondes de pause entre les modèles
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

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
            cv2.putText(image, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, result[0]['model'], (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




print(a)
print(result[0]['model']+","+result[0]['make'])