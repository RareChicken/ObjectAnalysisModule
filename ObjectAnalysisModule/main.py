import objanalyzer as oa

if __name__ == "__main__":
    objectAnalyzer = oa.ObjectAnalyzer('yolo_cpp_dll_no_gpu.dll', False, './data/dog.jpg')

    detections = objectAnalyzer.perform_detect()
    print(detections)

    objectAnalyzer.set_image_path('./data/eagle.jpg')
    detections = objectAnalyzer.perform_detect()
    print(detections)

    objectAnalyzer.set_image_path('./data/giraffe.jpg')
    detections = objectAnalyzer.perform_detect()
    print(detections)
