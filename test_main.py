import pytest
from main import *
import torch
import numpy as np 
import cv2 
'''
# Test pour la fonction getOutputsNames
def test_getOutputsNames():
    # Test avec un faux réseau
    class FakeNet:
        def getLayerNames(self):
            return ['layer1', 'layer2', 'layer3']

        def getUnconnectedOutLayers(self):
            return [1, 2]

    fake_net = FakeNet()
    assert getOutputsNames(fake_net) == ['layer1', 'layer2']

# Test pour la fonction preprocess_vehicle_region
def test_preprocess_vehicle_region():
    # Créer une image de test (un tableau numpy simulé)
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    # Appeler la fonction de prétraitement
    processed_image = preprocess_vehicle_region(image)

    # Vérifier le format de sortie
    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (1, 3, 224, 224)  # Vérifier la forme du tenseur

# Test pour la fonction matricule
def test_matricule():
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    outs = np.zeros((10, 7), dtype=np.float32)
    assert matricule(frame, outs) == ("Non detecte", "Inconnu")



# Test pour la classe VehicleClassifier
def test_VehicleClassifier():
    # Créer une instance de VehicleClassifier
    model = VehicleClassifier()

    # Vérifier si le modèle est bien initialisé
    assert isinstance(model, VehicleClassifier)

# Test pour la classe VehicleCounter
def test_VehicleCounter():
    # Créer une instance de VehicleCounter (vous pouvez fournir un chemin vidéo factice ou un fichier vidéo factice)
    vc = VehicleCounter("demo.mp4")

    # Vérifier si l'instance est bien initialisée
    assert isinstance(vc, VehicleCounter)

    # Vérifier si l'instance a bien tous les attributs attendus
    assert hasattr(vc, "broker_address")
    assert hasattr(vc, "broker_port")
    assert hasattr(vc, "topic")
    assert hasattr(vc, "mqtt_client")
    assert hasattr(vc, "tracker")
    assert hasattr(vc, "cam")
    assert hasattr(vc, "input_size")
    assert hasattr(vc, "confThreshold")
    assert hasattr(vc, "nmsThreshold")
    assert hasattr(vc, "classNames")
    assert hasattr(vc, "required_class_index")
    assert hasattr(vc, "net")
    assert hasattr(vc, "colors")

# Vous pouvez ajouter d'autres tests pour les autres fonctions et méthodes de votre script main.py
'''''

# Tester la fonction get_video_name
def get_video_name(video_path):                    
    return os.path.basename(video_path)

# Test pour la fonction getOutputsNames
def test_getOutputsNames():

    class FakeNet:
        def getLayerNames(self):
            return ['layer1', 'layer2', 'layer3']

        def getUnconnectedOutLayers(self):
            return [1, 2]

    fake_net = FakeNet()
    assert getOutputsNames(fake_net) == ['layer1', 'layer2']


# Tester la classe VehicleClassifier
def test_VehicleClassifier():
    model = VehicleClassifier(num_classes=2)
    assert isinstance(model, torch.nn.Module)

# Test pour la fonction preprocess_vehicle_region
def test_preprocess_vehicle_region():

    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    processed_image = preprocess_vehicle_region(image)

    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (1, 3, 224, 224)

# Tester la fonction matricule
def test_matricule_no_detection():
    """Tests if the function returns 'Non detecte' and 'Inconnu' when no plates are detected."""
    detected_text, detected_class = matricule(np.zeros((480, 640, 3)), [])
    assert detected_text == "Non detecte"
    assert detected_class == "Inconnu"


# Tester la classe VehicleCounter (partiellement, en supposant que le MQTT n'est pas disponible)
def test_VehicleCounter():
    vc = VehicleCounter("./demo.mp4")
    assert vc.video_name == "video.mp4"

# Tester la fonction send_video_name de la classe VehicleCounter
def test_send_video_name():
    vc = VehicleCounter("./demo.mp4")
    assert vc.send_video_name() == None 

# Tester la fonction publish_video_end_message de la classe VehicleCounter
def test_publish_video_end_message():
    vc = VehicleCounter("./demo.mp4")
    assert vc.publish_video_end_message() == None 
# Tester la fonction publish_json_to_mqtt de la classe VehicleCounter
def test_publish_json_to_mqtt():
    vc = VehicleCounter("./demo.mp4")
    assert vc.publish_json_to_mqtt({"test": "data"}) == None 

def test_VehicleCounter():
  
    vc = VehicleCounter("demo.mp4")

    # Vérifier si l'instance est bien initialisée
    assert isinstance(vc, VehicleCounter)

    # Vérifier si l'instance a bien tous les attributs attendus
    assert hasattr(vc, "broker_address")
    assert hasattr(vc, "broker_port")
    assert hasattr(vc, "topic")
    assert hasattr(vc, "mqtt_client")
    assert hasattr(vc, "tracker")
    assert hasattr(vc, "cam")
    assert hasattr(vc, "input_size")
    assert hasattr(vc, "confThreshold")
    assert hasattr(vc, "nmsThreshold")
    assert hasattr(vc, "classNames")
    assert hasattr(vc, "required_class_index")
    assert hasattr(vc, "net")
    assert hasattr(vc, "colors")

    