import sys

paths = ('../src/faceDetector/', 
         '../src/faceDetector/facenet/', 
         '../src/faceDetector/facenet/src/',
         '../src/faceDetector/facenet/src/align/',
         '../src/classifier/third_party/models/slim')

for path in paths :
    sys.path.insert(0, path)