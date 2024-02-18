VIDEO_PATH = './demo2.mp4'
INPUT_SIZE = 320

CONFIDENCE_THRESHOLD = 0.27
NMS_THRESHOLD = 0.2

FONT_COLOR = (0, 0, 255)
FONT_SIZE = 0.5
FONT_THICKNESS = 1

CLASSES_FILE = "./MODEL/coco.names"
REQUIRED_CLASS_INDEX = [2, 3, 5, 7]

MODEL_CONFIG = './MODEL/yolov4.cfg'
MODEL_WEIGHTS = './MODEL/yolov4.weights'


model_file = "./marque_model/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb"  # path to the car make and model classifier
label_file = "labels.txt"   # path to the text file, containing list with the supported makes and models
input_layer = "input_1"
output_layer = "softmax/Softmax"
classifier_input_size = (128, 128)  # input size of the classifier

modelConfiguration = "./matricule_model/darknet-yolov3.cfg"
modelWeights = "./matricule_model/model.weights"