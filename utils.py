import math
import os
import cv2
import numpy as np
from colorthief import ColorThief
from sklearn.cluster import MiniBatchKMeans
import webcolors



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

def find_center(x,y,w,h):
    return x + w//2, y+h//2

def count_vehicle(box_id, img, up_list, down_list, up_line_position, middle_line_position, down_line_position,
                  temp_up_list, temp_down_list):
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    ix, iy = center
    if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list:
            temp_up_list.append(id)
    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1
    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1
    cv2.circle(img, center, 2, (0, 0, 255), -1)

import requests
def detect_matricule(vehicle_region):
    api_key = '2a4ebc43c18c0c702388669a45e21a362a01d3f6'
    api_url = 'https://api.platerecognizer.com/v1/plate-reader/'
    headers = {'Authorization': f'Token {api_key}'}
    files = {'upload': ('image.jpg', cv2.imencode('.jpg', vehicle_region)[1])}

    response = requests.post(api_url, headers=headers, files=files)
    result = response.json()

    if 'results' in result and result['results']:
        s=result['results'][0]['plate']
        return s.upper()
    else:
        return 'Not Detected'
        

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76

def get_vehicle_color(vehicle_region):
    # Convertir la région du véhicule en format compatible avec K-Means
    vehicle_region = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2RGB)

    # Aplatir l'image en un tableau 2D
    pixels = vehicle_region.reshape((-1, 3))

    # Appliquer K-Means pour regrouper les couleurs
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(pixels)

    # Obtenir la couleur dominante du cluster
    dominant_color = kmeans.cluster_centers_[0]

    # Convertir les valeurs RGB en valeurs hexadécimales
    dominant_color_hex = rgb_to_hex(dominant_color)

    # Obtenir le nom de la couleur à partir de la valeur hexadécimale
    color_name = hex_to_name_custom(dominant_color_hex)

    return color_name

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def hex_to_name_custom(hex_color):
    # Liste de correspondances prédéfinies
    color_mapping = {
        "#FFA500": "Orange",
  "#FFA07A": "LightSalmon",
  "#FF7F50": "Coral",
  "#FF6347": "Tomato",
  "#FF4500": "OrangeRed",
  "#FF8C00": "DarkOrange",
  "#FFD700": "Gold",
  "#FFB6C1": "LightPink",
  "#FF69B4": "HotPink",
  "#FF1493": "DeepPink",
  "#FFC0CB": "Pink",
  "#FF00FF": "Magenta",
  "#FF00FF": "Fuchsia",
  "#DC143C": "Crimson",
  "#C71585": "MediumVioletRed",
  "#DA70D6": "Orchid",
  "#D8BFD8": "Thistle",
  "#DDA0DD": "Plum",
  "#EE82EE": "Violet",
  "#FFD700": "Fuchsia",
  "#000000": "Noir",
  "#FFFFFF": "Blanc",
  "#FFFFFF": "Blanc",
  "#FEFEFE": "Blanc cassé",
  "#FDFDFD": "Blanc navajo",
  "#FCFCFC": "Blanc linceul",
  "#FBFBFB": "Blanc ivoire",
  "#FAFAFA": "Blanc fantôme",
  "#F9F9F9": "Blanc hôpital",
  "#F8F8F8": "Blanc lait",
  "#F7F7F7": "Blanc étain",
  "#F6F6F6": "Blanc neige",
  "#F5F5F5": "Blanc fumée",
  "#F4F4F4": "Blanc eau",
  "#F3F3F3": "Blanc sel",
  "#F2F2F2": "Blanc porcelaine",
  "#F1F1F1": "Blanc étoile",
  "#F0F0F0": "Blanc laitier",
  "#EFEFEF": "Blanc papier",
  "#EEEEEE": "Blanc éclair",
  "#EDEDED": "Blanc glace",
  "#ECECEC": "Blanc lin",
  "#EBEBEB": "Blanc perle",
  "#EAEAEA": "Blanc émail",
  "#E9E9E9": "Blanc mercure",
  "#E8E8E8": "Blanc aile",
  "#E7E7E7": "Blanc vif",
  "#E6E6E6": "Blanc nuage",
  "#E5E5E5": "Blanc givre",
  "#E4E4E4": "Blanc soie",
  "#E3E3E3": "Blanc sable",
  "#E2E2E2": "Blanc crème",
  "#E1E1E1": "Blanc zircon",
  "#E0E0E0": "Blanc albatre",
  "#DFDFDF": "Blanc lumière",
  "#DEDEDE": "Blanc pur",
  "#DDDDDD": "Blanc papyrus",
  "#DCDCDC": "Blanc poudre",
  "#DBDBDB": "Blanc écume",
  "#DADADA": "Blanc ivoire",
  "#D9D9D9": "Blanc diamant",
  "#D8D8D8": "Blanc glacial",
  "#D7D7D7": "Blanc lustré",
  "#D6D6D6": "Blanc banquise",
  "#D5D5D5": "Blanc zéphyr",
  "#D4D4D4": "Blanc argent",
  "#D3D3D3": "Blanc gris",
  "#D2D2D2": "Blanc lunaire",
  "#D1D1D1": "Blanc porche",
  "#D0D0D0": "Blanc minéral",
  "#CFCFCF": "Blanc nickel",
  "#CECECE": "Blanc os",
  "#CDCDCD": "Blanc merisier",
  "#CCCCCC": "Blanc poussière",
  "#CBCBCB": "Blanc pampre",
  "#CACACA": "Blanc percale",
  "#C9C9C9": "Blanc silex",
  "#C8C8C8": "Blanc coquille",
  "#C7C7C7": "Blanc pâle",
  "#C6C6C6": "Blanc quartz",
  "#C5C5C5": "Blanc flocon",
  "#C4C4C4": "Blanc orage",
  "#C3C3C3": "Blanc ouate",
  "#C2C2C2": "Blanc linceul",
  "#C1C1C1": "Blanc kashmir",
  "#C0C0C0": "Argent",
  "#BFBFBF": "Gris clair",
  "#BEBEBE": "Gris souris",
  "#BDBDBD": "Gris perle",
  "#BCBCBC": "Gris éclair",
  "#BBBBBB": "Gris nuage",
  "#BABABA": "Gris givre",
  "#B8B8B8": "Gris métal",
  "#B7B7B7": "Gris argent",
  "#B6B6B6": "Gris acier",
  "#B5B5B5": "Gris tourterelle",
  "#B4B4B4": "Gris plomb",
  "#B3B3B3": "Gris ardoise",
  "#B2B2B2": "Gris perle",
  "#B1B1B1": "Gris rocher",
  "#B0B0B0": "Gris étain",
  "#AFAFAF": "Gris fer",
  "#AEAEAE": "Gris métallisé",
  "#ADADAD": "Gris alu",
  "#ACACAC": "Gris béton",
  "#ABABAB": "Gris souris",
  "#AAAAAA": "Gris",
  "#A9A9A9": "Gris sombre",
  "#A8A8A8": "Gris foncé",
  "#A7A7A7": "Gris anthracite",
  "#A6A6A6": "Gris fer forgé",
  "#A5A5A5": "Gris plomb",
  "#A4A4A4": "Gris ardoise",
  "#A3A3A3": "Gris fer",
  "#A2A2A2": "Gris graphite",
  "#A1A1A1": "Gris fer",
  "#A0A0A0": "Gris",
  "#9F9F9F": "Gris clair",
  "#9E9E9E": "Gris souris",
  "#9D9D9D": "Gris perle",
  "#9C9C9C": "Gris éclair",
  "#9B9B9B": "Gris nuage",
  "#9A9A9A": "Gris givre",
  "#999999": "Gris fumée",
  "#FFFF00": "yellow",
  "#FFFF33": "yellow",
  "#FFFF66": "yellow",
  "#FFFF99": "yellow",
  "#FFFFCC": "yellow",
  "#FFFC00": "yellow",
  "#FFFC33": "yellow",
  "#FFFC66": "yellow",
  "#FFFC99": "yellow",
  "#FFCCCC": "yellow",
  "#FFCC00": "yellow",
  "#FFCC33": "yellow",
  "#FFCC66": "yellow",
  "#FFCC99": "yellow",
  "#FFCCCC": "yellow",
  "#FFD700": "yellow",
  "#FFD733": "yellow",
  "#FFD766": "yellow",
  "#FFD799": "yellow",
  "#FFD7CC": "yellow",
  "#FFDB00": "yellow",
  "#FFDB33": "yellow",
  "#FFDB66": "yellow",
  "#FFDB99": "yellow",
  "#FFDBCC": "yellow",
  "#FFDE00": "yellow",
  "#FFDE33": "yellow",
  "#FFDE66": "yellow",
  "#FFDE99": "yellow",
  "#FFDECC": "yellow",
  "#FFE700": "yellow",
  "#FFE733": "yellow",
  "#FFE766": "yellow",
  "#FFE799": "yellow",
  "#FFE7CC": "yellow",
  "#FFEB00": "yellow",
  "#FFEB33": "yellow",
  "#FFEB66": "yellow",
  "#FFEB99": "yellow",
  "#FFEBCC": "yellow",
  "#FFEE00": "yellow",
  "#FFEE33": "yellow",
  "#FFEE66": "yellow",
  "#FFEE99": "yellow",
  "#FFEECC": "yellow",
  "#FFF200": "yellow",
  "#FFF233": "yellow",
  "#FFF266": "yellow",
  "#FFF299": "yellow",
  "#FFF2CC": "yellow",
  "#FFF600": "yellow",
  "#FFF633": "yellow",
  "#FFF666": "yellow",
  "#FFF699": "yellow",
  "#FFF6CC": "yellow",
  "#FFFA00": "yellow",
  "#FFFA33": "yellow",
  "#FFFA66": "yellow",
  "#FFFA99": "yellow",
  "#FFFACC": "yellow",
  "#FFFB00": "yellow",
  "#FFFB33": "yellow",
  "#FFFB66": "yellow",
  "#FFFB99": "yellow",
  "#FFFCCC": "yellow",
  "#FFFD00": "yellow",
  "#FFFD33": "yellow",
  "#FFFD66": "yellow",
  "#FFFD99": "yellow",
  "#FFFDC": "yellow",
  "#FFFF00": "yellow",
  "#FFFF33": "yellow",
  "#FFFF66": "yellow",
  "#FFFF99": "yellow",
  "#FFFFCC": "yellow",
  "#FFFFF": "yellow",
  "#00FF00": "Lime",
  "#00FA9A": "MediumSpringGreen",
  "#00FF7F": "SpringGreen",
  "#7CFC00": "LawnGreen",
  "#32CD32": "LimeGreen",
  "#ADFF2F": "GreenYellow",
  "#228B22": "ForestGreen",
  "#008000": "Green",
  "#006400": "DarkGreen",
  "#9ACD32": "YellowGreen",
  "#6B8E23": "OliveDrab",
  "#8FBC8F": "DarkSeaGreen",
  "#20B2AA": "LightSeaGreen",
  "#008080": "Teal",
  "#2E8B57": "SeaGreen",
  "#3CB371": "MediumSeaGreen",
  "#66CDAA": "MediumAquamarine",
  "#00CED1": "DarkTurquoise",
  "#00FFFF": "Aqua/Cyan",
  "#00FFFF": "Turquoise",
  "#E0FFFF": "LightCyan",
  "#7FFFD4": "Aquamarine",
  "#F0FFF0": "Honeydew",
  "#5F9EA0": "CadetBlue",
  "#4682B4": "SteelBlue",
  "#B0C4DE": "LightSteelBlue",
  "#ADD8E6": "LightBlue",
  "#87CEFA": "LightSkyBlue",
  "#87CEEB": "SkyBlue",
  "#00BFFF": "DeepSkyBlue",
  "#1E90FF": "DodgerBlue",
  "#6495ED": "CornflowerBlue",
  "#4169E1": "RoyalBlue",
  "#0000FF": "Blue",
  "#0000CD": "MediumBlue",
  "#00008B": "DarkBlue",
  "#000080": "Navy",
  "#191970": "MidnightBlue",
  "#8A2BE2": "BlueViolet",
  "#9400D3": "DarkViolet",
  "#9932CC": "DarkOrchid",
  "#8B008B": "DarkMagenta",
  "#800080": "Purple",
  "#4B0082": "Indigo",
  "#6A5ACD": "SlateBlue",
  "#483D8B": "DarkSlateBlue",
  "#7B68EE": "MediumSlateBlue",
  "#9370DB": "MediumPurple",
  "#8B4513": "SaddleBrown",
  "#D2691E": "Chocolate",
  "#CD5C5C": "IndianRed",
  "#DC143C": "Crimson",
  "#FF4500": "OrangeRed",
  "#FF6347": "Tomato",
  "#FF7F50": "Coral",
  "#FF8C00": "DarkOrange",
  "#FFA07A": "LightSalmon",
  "#FFA500": "Orange",
  "#FFD700": "Gold",
  "#FFB90F": "DarkGoldenrod",
  "#FFDAB9": "PeachPuff",
  "#FFE4B5": "Moccasin",
  "#FFEFD5": "PapayaWhip",
  "#FFFACD": "LemonChiffon",
  "#FAFAD2": "LightGoldenrodYellow",
  "#FFFFE0": "LightYellow",
  "#FFFF00": "Yellow",
  "#FFD700": "Gold",
  "#FFFAF0": "FloralWhite",
  "#FDF5E6": "OldLace",
  "#FFE4E1": "MistyRose",
  "#FAEBD7": "AntiqueWhite",
  "#FFF8DC": "Cornsilk",
  "#FFF5EE": "SeaShell",
  "#F5F5DC": "Beige",
  "#F5DEB3": "Wheat",
  "#F4A460": "SandyBrown",
  "#D2B48C": "Tan",
  "#DEB887": "BurlyWood",
  "#FFE4C4": "Bisque",
  "#FF8C00": "DarkOrange",
  "#FAF0E6": "Linen",
  "#CD853F": "Peru",
  "#8B4513": "SaddleBrown",
  "#A52A2A": "Brown",
  "#800000": "Maroon",
  "#0000FF": "Bleu",
  "#0000CD": "Bleu moyen",
  "#00008B": "Bleu foncé",
  "#000080": "Bleu marine",
  "#0000A0": "Bleu de minuit",
  "#4169E1": "Bleu roi",
  "#87CEEB": "Bleu ciel",
  "#87CEFA": "Bleu lueur",
  "#00BFFF": "Bleu azur",
  "#1E88E5": "Bleu attirance",
  "#1976D2": "Bleu indigo",
  "#1C94E0": "Bleu persan",
  "#0D47A1": "Bleu de Prusse",
  "#1565C0": "Bleu ardoise",
  "#42A5F5": "Bleu clair",
  "#64B5F6": "Bleu glacial",
  "#90CAF9": "Bleu de l'air",
  "#BBDEFB": "Bleu Alice",
  "#2196F3": "Bleu électrique",
  "#0D47A1": "Bleu profond",
  "#0F7DC2": "Bleu outremer",
  "#1569C7": "Bleu cobalt",
  "#2E8B57": "Bleu mer",
  "#008080": "Bleu sarcelle",
  "#008B8B": "Bleu paon",
  "#00CED1": "Bleu turquoise",
  "#00FFFF": "Bleu cyan",
  "#E0FFFF": "Bleu turquoise pâle",
  "#82CAFA": "Bleu poudre",
  "#AFEEEE": "Bleu turquoise clair",
  "#00FA9A": "Bleu-vert",
  "#00FF7F": "Bleu de printemps",
  "#2E8B57": "Vert mer",
  "#008000": "Vert",
  "#8FBC8F": "Vert lavande",
  "#90EE90": "Vert pâle",
  "#98FB98": "Vert menthe",
  "#00FA9A": "Vert moyen",
  "#00FF7F": "Vert printemps",
  "#3CB371": "Vert marine",
  "#2E8B57": "Vert mer",
  "#00CED1": "Bleu turquoise",
  "#20B2AA": "Bleu acier clair",
  "#66CDAA": "Bleu turquoise moyen",
  "#7FFFD4": "Bleu turquoise clair",
  "#AFEEEE": "Bleu turquoise clair",
  "#00FFFF": "Bleu cyan",
  "#00FFFF": "Bleu cyan",
  "#E0FFFF": "Bleu turquoise pâle",
  "#E0FFFF": "Bleu turquoise pâle",
  "#00CED1": "Bleu turquoise",
  "#5F9EA0": "Bleu turquoise foncé",
  "#008B8B": "Bleu paon",
  "#008B8B": "Bleu paon",
  "#008080": "Bleu sarcelle",
  "#48D1CC": "Bleu sarcelle médium",
  "#20B2AA": "Bleu sarcelle clair",
  "#40E0D0": "Turquoise médium",
  "#00FA9A": "Vert moyen",
  "#7FFF00": "Vert chartreuse",
  "#32CD32": "Vert lime vif",
  "#00FF00": "Vert lime",
  "#008000": "Vert",
  "#2E8B57": "Vert mer",
  "#008B8B": "Bleu paon",
  "#008B8B": "Bleu paon",
  "#00CED1": "Bleu turquoise",
  "#00FFFF": "Bleu cyan",
  "#E0FFFF": "Bleu turquoise pâle",
  "#82CAFA": "Bleu poudre",
  "#AFEEEE": "Bleu turquoise clair",
  "#00FA9A": "Bleu-vert",
  "#00FF7F": "Bleu de printemps",
  "#2E8B57": "Vert mer",
  "#008000": "Vert",
  "#00FF00": "Vert lime",
  "#7FFF00": "Vert chartreuse",
  "#32CD32": "Vert lime vif",
  "#228B22": "Vert forêt",
  "#008000": "Vert foncé",
  "#3CB371": "Vert marine",
  "#2E8B57": "Vert mer",
  "#00CED1": "Bleu turquoise",
  "#20B2AA": "Bleu acier clair",
  "#66CDAA": "Bleu turquoise moyen",
  "#7FFFD4": "Bleu turquoise clair",
  "#AFEEEE": "Bleu turquoise clair",
  "#00FFFF": "Bleu cyan",
  "#E0FFFF": "Bleu turquoise pâle",
  "#00CED1": "Bleu turquoise",
  "#5F9EA0": "Bleu turquoise foncé",
  "#008B8B": "Bleu paon",
  "#008080": "Bleu sarcelle",
  "#48D1CC": "Bleu sarcelle médium",
  "#20B2AA": "Bleu sarcelle clair",
  "#40E0D0": "Turquoise médium",
  "#FF0000": "Rouge pur",
  "#DC143C": "Rouge cramoisi",
  "#FF4500": "Rouge orangé",
  "#FF6347": "Rouge corail",
  "#FF7F50": "Rouge saumon",
  "#FF8C00": "Rouge foncé",
  "#FFA07A": "Rouge clair",
  "#FF0000": "Rouge vif",
  "#CD5C5C": "Rouge indien",
  "#8B0000": "Rouge sombre",
  "#B22222": "Rouge brique",
  "#A52A2A": "Rouge brun",
  "#FF69B4": "Rouge orchidée",
  "#FF1493": "Rouge profond",
  "#DB7093": "Rouge pâle",
  "#FFA500": "Rouge orangé",
  "#FFD700": "Rouge doré",
  "#D2691E": "Rouge cannelle",
  "#CD853F": "Rouge brun clair",
  "#A52A2A": "Rouge acajou",
  "#8B4513": "Rouge rouille",
  "#800000": "Rouge pourpre",
  "#8B008B": "Rouge violet",
  "#9932CC": "Rouge foncé",
  "#9400D3": "Rouge violet profond",
  "#800080": "Rouge violet foncé",
  "#9370DB": "Rouge violet moyen",
  "#4B0082": "Rouge indigo",
  "#6A5ACD": "Rouge lavande",
  "#483D8B": "Rouge bleu foncé",
  "#0000FF": "Rouge bleu",
  "#0000CD": "Rouge médium",
  "#00008B": "Rouge foncé",
  "#000080": "Rouge marine",
  "#4169E1": "Rouge royal",
  "#6495ED": "Rouge bleu acier",
  "#1E90FF": "Rouge dodger",
  "#00BFFF": "Rouge azur",
  "#87CEEB": "bleu ciel",
  "#87CEFA": "Rouge lumière bleue",
  "#ADD8E6": "Rouge clair",
  "#B0E0E6": "Rouge poudre d'argent",
  "#AFEEEE": "Rouge turquoise pâle",
  "#00CED1": "Rouge turquoise foncé",
  "#48D1CC": "Rouge turquoise médium",
  "#20B2AA": "Rouge turquoise clair",
  "#40E0D0": "Rouge turquoise",
  "#00FA9A": "Rouge vert moyen",
  "#00FF7F": "Rouge printemps",
  "#3CB371": "Rouge foncé",
  "#2E8B57": "Rouge marin",
  "#228B22": "Rouge forêt",
  "#008000": "Rouge vert",
  "#006400": "Rouge foncé",
  "#9ACD32": "Rouge jaune-vert",
  "#6B8E23": "Rouge olivâtre",
  "#808000": "Rouge olive",
  "#8FBC8F": "Rouge mer",
  "#20B2AA": "Rouge lumière marine",
  "#00FFFF": "Rouge turquoise",
  "#00CED1": "Rouge turquoise foncé",
  "#00FA9A": "Rouge turquoise médium",
  "#008080": "Rouge cyan foncé",
  "#008B8B": "Rouge cyan",
  "#00BFFF": "Rouge azur",
  "#1E90FF": "Rouge dodger",
  "#000080": "Rouge bleu marine",
  "#00008B": "Rouge foncé",
  "#0000CD": "Rouge médium",
  "#0000FF": "Rouge bleu",
  "#8A2BE2": "Rouge bleu violet",
  "#9400D3": "Rouge violet profond",
  "#9932CC": "Rouge violet foncé",
  "#800080": "Rouge violet",
  "#4B0082": "Rouge indigo",
  "#6A5ACD": "Rouge lavande",
  "#9370DB": "Rouge violet moyen",
  "#DDA0DD": "Rouge violet clair",
  "#D8BFD8": "Rouge thistle",
  "#E6E6FA": "Rouge lavande pâle",
  "#F8F8FF": "Rouge blanc",
  "#FFFFFF": "Rouge pur"
}


    # Recherchez la correspondance exacte dans le dictionnaire
    exact_match = color_mapping.get(hex_color)
    if exact_match:
        return exact_match

    # Si aucune correspondance exacte, trouvez la correspondance la plus proche
    closest_color = find_closest_color_lab(hex_color, list(color_mapping.keys()))
    return color_mapping.get(closest_color, "unknown")

def find_closest_color_lab(target_color, color_list):
    # Convertir la couleur cible en espace colorimétrique LAB
    target_color_lab = rgb2lab(np.array([[int(target_color[i:i+2], 16) for i in (1, 3, 5)]]))

    # Convertir les couleurs de la liste en espace colorimétrique LAB
    color_list_lab = [rgb2lab(np.array([[int(color[i:i+2], 16) for i in (1, 3, 5)]])) for color in color_list]

    # Calculer la distance euclidienne entre la couleur cible et chaque couleur dans la liste
    distances = [deltaE_cie76(target_color_lab[0], color_lab[0]) for color_lab in color_list_lab]

    # Trouver l'index de la couleur la plus proche
    closest_color_index = np.argmin(distances)

    return color_list[closest_color_index]

# Assurez-vous d'importer les bibliothèques nécessaires au début de votre fichier


def postProcess(outputs, img, colors, classNames, confThreshold, nmsThreshold, required_class_index, tracker,
                up_list, down_list, up_line_position, middle_line_position, down_line_position, temp_up_list,
                temp_down_list, is_double_clicked, font_color, font_size, font_thickness):
    detected_vehicles = []  # Nouvelle liste pour stocker les véhicules détectés
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
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
        vehicle_colors = get_vehicle_color(vehicle_region)  # Obtenir les noms de couleurs dominantes du véhicule

        
    
        detected_vehicles.append({
            "name": name,
            "confidence": int(confidence_scores[i]*100),
            "position": (x, y, w, h),
            "colors": vehicle_colors,
            
        })

        if vehicle_colors is not None:
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}% - Colors: {", ".join(map(str, vehicle_colors))}',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)
        else:
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)
       

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Modifiez cette ligne pour renvoyer la liste detection
    return detected_vehicles
