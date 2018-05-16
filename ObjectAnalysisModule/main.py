import objanalyzer as oa
import cv2

def kaze_match(im1_path, im2_path):
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    print('keypoints: {}, descriptors: {}'.format(len(kps1), descs1.shape))
    print('keypoints: {}, descriptors: {}'.format(len(kps2), descs2.shape))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1, descs2, k=2)

    good = []
    for (m, n) in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    im3 = cv2.drawMatchesKnn(im1, kps1, im2, kps2, good, None, flags=2)
    cv2.imshow('AKAZE matching', im3)
    cv2.waitKey(0)

if __name__ == "__main__":
    #objectAnalyzer = oa.ObjectAnalyzer('yolo_cpp_dll_no_gpu.dll', False, './data/dog.jpg')

    #detections = objectAnalyzer.perform_detect()
    #print(detections)

    #objectAnalyzer.set_image_path('./data/eagle.jpg')
    #detections = objectAnalyzer.perform_detect()
    #print(detections)

    #objectAnalyzer.set_image_path('./data/giraffe.jpg')
    #detections = objectAnalyzer.perform_detect()
    #print(detections)

    kaze_match('./data/giraffe.jpg', './data/giraffeRotate.jpg')
