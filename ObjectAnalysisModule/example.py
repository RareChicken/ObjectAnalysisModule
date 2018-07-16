from object_searcher import *
from darknet_yolo import ObjFlag

def main():
    objsearcher = ObjectSearcher()
    result = objsearcher.search('data/test.mp4', ObjFlag.CAR | ObjFlag.PERSON)
    print(result)

if __name__ == '__main__':
    main()
