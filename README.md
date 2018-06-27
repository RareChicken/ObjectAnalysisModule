# ObjectAnalysisModule
객체 분석 모듈

# Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO_DEEP_SORT 

```
   wget https://pjreddie.com/media/files/yolov3.weights
   python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
   python demo.py
```

# Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are needed to run the tracker:

    NumPy
    sklean
    OpenCV

Additionally, feature generation requires TensorFlow-1.4.0.
