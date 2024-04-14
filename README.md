# C++ Visual object detection with YOLOv8, Pytorch and OpenCV
This project is a C++ implementation of a visual object detection system using the YOLOv8 model, Pytorch, and OpenCV. 
The YOLOv8 model is loaded in the torchscript format. The system predicts objects in an input frame captured from a webcam.
The project includes a YoloV8detect class that handles the loading of the model and the prediction of objects. 
The class also includes methods for non-maximum suppression and color generation for the bounding boxes. 
The main application sets up the video capture, runs the object detection on each frame, and draws the bounding boxes on the detected objects.

## Requirements
I used CLion as the IDE and as environment the docker container from this [repository](https://github.com/sleepingwithshoes/torchopencv).
All dependencies are installed in the docker container.
Follow the instructions in the respective to set up the environment.

## Usage
### Generate the torchscript model (optional)
This repository contains converted torchscript models of the YOLOv8 model. 
But if you want to convert the model yourself, you can follow these steps.

Install the ultralytics package:
```bash
pip install ultralytics
```

To generate the torchscript model from the YOLOv8 model, you need only to execute this short Python script:
```python
from ultralytics import YOLO

# load the model
model = YOLO("yolov8n.pt", task="detect")

#Export the model to torchscript format with input size 640x640
success = model.export(format="torchscript", imgsz=[640,640])
```
This script will generate a file named `yolov8n.torchscript` in the same directory where the script is executed.

### Use the C++ application
Pass the path to the torchscript model, class names, and the input size to the YoloV8detect constructor.
```cpp 
 YoloV8detect yoloV8detect("../../model/yolov8/yolov8n640.torchscript","../../model/classes/coco.names",{640,640},device);
```
Select the webcam device (0 is mostly the internal webcam of your laptop) to capture the video stream.
```cpp
cap = cv::VideoCapture cap(0);
```
...and build and run the application.

## Notes
My motivation of this project was to learn how to use YOLO models in C++ applications. 
Feel free to use the code and modify it to your needs.

**Happy coding!**