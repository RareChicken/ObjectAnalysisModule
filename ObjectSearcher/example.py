from object_searcher import *
from darknet_yolo import ObjFlag
from darknet_yolo import YOLO

def main(yolo):
    objsearcher = ObjectSearcher(yolo)
    result = objsearcher.search('data/test.mp4', ObjFlag.CAR.value)
    print(result)

if __name__ == '__main__':
    main(YOLO())
