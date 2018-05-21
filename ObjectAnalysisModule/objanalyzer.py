from ctypes import *
import os

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class ObjectAnalyzer(object):
    def __init__(self, dllName, isGPUDLL, imagePath,
                 thresh= 0.25, configPath= "./cfg/yolov3.cfg",
                 weightPath= "yolov3.weights", metaPath= "./data/coco.data",
                 initOnly= False, netMain= None, metaMain= None, altNames= None):
        """
        Parameters
        ----------------
        imagePath: str
            Path to the image to evaluate. Raises ValueError if not found

        thresh: float (default= 0.25)
            The detection threshold

        configPath: str
            Path to the configuration file. Raises ValueError if not found

        weightPath: str
            Path to the weights file. Raises ValueError if not found

        metaPath: str
            Path to the data file. Raises ValueError if not found

        showImage: bool (default= True)
            Compute (and show) bounding boxes. Changes return.

        makeImageOnly: bool (default= False)
            If showImage is True, this won't actually *show* the image, but will create the array and return it.

        initOnly: bool (default= False)
            Only initialize globals. Don't actually run a prediction.
        """

        self.imagePath = imagePath
        self.thresh = thresh
        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath
        self.initOnly = initOnly
        self.netMain = netMain
        self.metaMain = metaMain
        self.altNames = altNames

        set_dll(dllName, isGPUDLL)

        if self.netMain is None:
            self.netMain = self.load_net(configPath.encode("ascii"), weightPath.encode("ascii"), 0)
        if self.metaMain is None:
            self.metaMain = self.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            # In Python 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def perform_detect(self):
        """1
        Convenience function to handle the detection and returns of objects.

        Displaying bounding boxes requires libraries scikit-image and numpy

        Returns
        ----------------------

        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.
        """
        assert 0 < self.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
        if self.initOnly:
            print("Initialized detector")
            return None
        if not os.path.exists(self.imagePath):
            raise ValueError("Invalid image path `"+os.path.abspath(self.imagePath)+"`")
        # Do the detection
        detections = self._detect()
        self.detections = detections

        return detections

    def _detect(self, hier_thresh=.5, nms=.45, debug= False):
        """
        Performs the meat of the detection
        """
        #pylint: disable= C0321
        im = self.load_image(self.imagePath.encode('ascii'), 0, 0)
        if debug: print("Loaded image")
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self.predict_image(self.netMain, im)
        if debug: print("did prediction")
        dets = self.get_network_boxes(self.netMain, im.w, im.h, self.thresh, hier_thresh, None, 0, pnum, 1)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.metaMain.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on "+str(j)+" of "+str(num))
            if debug: print("Classes: "+str(self.metaMain), self.metaMain.classes, self.metaMain.names)
            for i in range(self.metaMain.classes):
                if debug: print("Class-ranging on "+str(i)+" of "+str(self.metaMain.classes)+"= "+str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = self.metaMain.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self.free_image(im)
        if debug: print("freed image")
        self.free_detections(dets, num)
        if debug: print("freed detections")
        return res

    def show_image(self, detections, makeImageOnly= False):
        """
        Return
        ----------------------

        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
        """

        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(self.imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
            return detections
        except Exception as e:
            print("Unable to show image: "+str(e))
            return detections

    def set_dll(self, dllName, isGPUDLL):
        lib = CDLL(dllName, RTLD_GLOBAL)
        ########## dll 함수 형식 설정 ##########
        if isGPUDLL:
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]

        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        self.predict = lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.make_image = lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)
        ####################
        self.lib = lib

    def set_image_path(self, imagePath):
        self.imagePath = imagePath

    def get_detections(self):
        return self.detections