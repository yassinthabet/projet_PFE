import pytest
import cv2
import numpy as np
import torch
from utils import *
# Tests pour load_checkpoint()

def test_load_checkpoint_success():
    model = load_checkpoint("brand.pth")
    assert model is not None


# Tests pour tansf_image()

def test_tansf_image():
    image = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)  # Créer une image de test
    transformed_image = tansf_image(image)
    assert transformed_image.shape == (1, 3, 224, 224)


# Tests pour EuclideanDistTracker.update()

def test_euclidean_dist_tracker_update():
    tracker = EuclideanDistTracker()
    objects_rect = [(10, 20, 40, 50, 0), (50, 60, 30, 40, 1), (80, 90, 25, 35, 2)]
    objects_bbs_ids = tracker.update(objects_rect)
    assert objects_bbs_ids == [[10, 20, 40, 50, 0, 0], [50, 60, 30, 40, 1, 1], [80, 90, 25, 35, 2, 2]]


# Tests pour get_color_name()
def test_get_color_name():
    # Test cases for black color (v < 50)
    assert get_color_name((0, 0, 40)) == "Noir"
    assert get_color_name((0, 0, 20)) == "Noir"

    # Test cases for white color (s < 50)
    assert get_color_name((0, 30, 255)) == "Blanc"
    assert get_color_name((0, 10, 255)) == "Blanc"

    # Test cases for different hue ranges
    assert get_color_name((5, 200, 200)) == "Rouge"
    assert get_color_name((30, 200, 200)) == "Jaune"
    assert get_color_name((70, 200, 200)) == "Vert"
    assert get_color_name((105, 200, 200)) == "Cyan"
    assert get_color_name((135, 200, 200)) == "Bleu"
    assert get_color_name((157, 200, 200)) == "Magenta"
    assert get_color_name((15, 200, 200)) == "Jaune"  # Corrected comparison

def test_get_color_with_valid_input():
    # Créer une région de véhicule simulée avec une couleur spécifique
    vehicle_region = np.zeros((100, 100, 3), dtype=np.uint8)
    vehicle_region[:, :] = [0, 255, 255]  # Couleur jaune en HSV

    # Définir les valeurs HSV de la couleur à détecter (jaune)
    v1_min, v2_min, v3_min = 20, 100, 100
    v1_max, v2_max, v3_max = 30, 255, 255

    # Appeler la fonction à tester
    color_name = get_color(vehicle_region, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max)

    # Vérifier si la couleur détectée est correcte
    assert color_name == "Jaune"

def test_get_color_with_invalid_input():
    # Créer une région de véhicule vide
    vehicle_region = None

    # Définir les valeurs HSV arbitraires
    v1_min, v2_min, v3_min = 0, 0, 0
    v1_max, v2_max, v3_max = 255, 255, 255

    # Appeler la fonction à tester avec une entrée invalide
    color_name = get_color(vehicle_region, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max)

    # Vérifier si la fonction retourne None pour une entrée invalide
    assert color_name is None

# Tests pour dist_calculator()

def test_dist_calculator():
    # Tester le calcul de la distance
    distance = dist_calculator(50, 100)
    assert isinstance(distance, float)


# Tests pour find_closest_vehicle()

def test_find_closest_vehicle():
    vehicles = [{"position": (100, 100, 100, 100)}]
    assert find_closest_vehicle(vehicles, 640) == vehicles[0]
def test_postProcess():
    # Créer des données de test
    outputs = [np.random.rand(10, 7), np.random.rand(10, 7)]
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    colors = ["Rouge", "Bleu"]
    classNames = ["voiture", "camion"]
    confThreshold = 0.5
    nmsThreshold = 0.5
    required_class_index = [0, 1]
    tracker = EuclideanDistTracker()
    # Tester le post-traitement des résultats
    result = postProcess(outputs, img, colors, classNames, confThreshold, nmsThreshold, required_class_index, tracker)
    assert result is not None

if __name__ == "__main__":
    pytest.main()
