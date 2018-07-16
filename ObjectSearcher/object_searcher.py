from __future__ import division, print_function, absolute_import

import os
import shutil
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
# from yolo import YOLO
from darknet_yolo import YOLO
from darknet_yolo import array_to_image

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

class ObjectSearcher:
    def __init__(self):
        self.nms_max_overlap = 1.0

        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        self.yolo = YOLO()

    def _getTracker(self):
        max_cosine_distance = 0.3
        nn_budget = None

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        return Tracker(metric)

    def search(self, video_path, flag= 0):
        result = []

        if flag == 0:
            return result

        video_capture = cv2.VideoCapture(video_path)

        tracker = _getTracker()

        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        frame_index = -1

        last_id = 0
        fps = 0.0
        while True:
            ret, original_frame = video_capture.read()
            ret, frame = video_capture.read()  # frame shape 640*480*3
            frame_index = frame_index + 1
            if ret != True:
                break

            # image = Image.fromarray(frame)
            image, arr = array_to_image(frame)
            dets = self.yolo.detect_image(image, flag)

            # print("box_num",len(boxs))
            boxes = [det[1] for det in dets]
            features = self.encoder(frame, boxes)
        
            # score to 1.0 here).
            detections = [Detection(det[0], det[1], 1.0, feature) for det, feature in zip(dets, features)]
        
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
        
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update > 1 :
                    continue
                bbox = track.to_tlbr()
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                cv2.rectangle(frame, (x, y), (w, h), (255,255,255), 2)
                cv2.putText(frame, str(track.track_id), (x, y), 0, 5e-3 * 200, (0,255,0), 2)

                if track.track_id > last_id:
                    image_trim = original_frame[y:h, x:w]
                    result.append({
                        id: track.track_id,
                        tag: track.clazz,
                        image: image_trim,
                        frame_idx: frame_index + 1
                    })

            last_id = tracker._next_id - 1;

            for det in detections:
                bbox = det.to_tlbr()
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                cv2.rectangle(frame, (x, y), (w, h), (255,0,0), 2)
                cv2.putText(frame, det.clazz, (x, y), 0, 5e-3 * 200, (0,0,255), 2)

            # save a frame
            out.write(frame)
            list_file.write(str(frame_index)+' ')
            if len(boxes) != 0:
                for i in range(0,len(boxes)):
                    list_file.write(str(boxes[i][0])+' '+str(boxes[i][1])+ ' '+str(boxes[i][2])+' '+str(boxes[i][3])+' ')
            list_file.write('\n')

        video_capture.release()
        out.release()

        return result
