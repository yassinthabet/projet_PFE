
# 🚗 Vehicle Detection, Classification, and Counting using OpenCV

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📁 Repository Contents

- 📜 `main.py`: Launches the vehicle counter.
- 🛠 `utils.py`: Houses utility functions and classes for vehicle operations.
- ⚙️ `config.py`: Manages parameters such as model paths, video input, and display settings.

- 📦 `MODEL/`: (`yolov4.weights` missing in this repo - [download](https://drive.google.com/file/d/1qTdvxKKP4K9u5GJrffufSx6cpR1AmLoz/view?usp=sharing))
  - 🧠 `yolov4.cfg`: YOLO model config.
  - 🔖 `coco.names`: Recognizable classes by the model.



## 🚀 Usage

1. 🔗 Clone this repository.
2. 📦 Install dependencies:

   ```shell 
   pip install -r requirements.txt
   ```
   
3. ⚙️ Adjust paths and parameters in config.py.
4. 🏃‍♂️ Execute:
   ```shell
   python main.py video.mp4
   ```
5. 🖱 In the new window, double-click to position the counting line. Watch as vehicle stats get tallied in real-time!

---

## ⚙️ Configurations Parameters
- `INPUT_SIZE`: YOLO model's desired input dimensions.
- `CONFIDENCE_THRESHOLD`: Desired confidence level for detections.
- `NMS_THRESHOLD`: Non-max suppression's threshold.
- `FONT_COLOR`, `FONT_SIZE`, `FONT_THICKNESS`: Style the on-screen text.
- `CLASSES_FILE`: YOLO's class definitions file path.
- `REQUIRED_CLASS_INDEX`: Indices of classes you wish to monitor.
- `MODEL_CONFIG`: YOLO's config file path.
- `MODEL_WEIGHTS`: YOLO's weight file path (download separately).

> 🛠 Tweak these in config.py to fit your scenario.



yolov3.weights must be downloaded from https://pjreddie.com/media/files/yolov3.weights and saved in folder yolo-coco

