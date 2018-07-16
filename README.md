# ObjectAnalysisModule
객체 분석 모듈

# Quick Start

1. YOLOv3 weights 파일을 여기서 다운. [YOLO website](http://pjreddie.com/darknet/yolo/).
2. model_data 폴더 내부에 YOLOv3 weights 파일 주입.
3. ObjectAnalysisModule.sln 열어서 실행.

# CUDA
cuda 9.0 버전을 사용.

### CUDA가 설치되어 있고 GPU를 사용하려는 경우
시스템 환경 변수에 CUDA9.0 경로가 담겨있는 CUDA_HOME변수 추가.
### 무조건 CPU를 사용하려는 경우
시스템 환경 변수에 1 혹은 true 혹은 yes 혹은 on이 들어간 FORCE_CPU변수 추가.

# Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are needed to run the tracker:

    NumPy
    sklean
    OpenCV
