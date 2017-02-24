import facedetect
import cv2
import sys

imagePath = sys.argv[1]
cascPath = sys.argv[2]
fileToSaveImage = sys.argv[3]

face = facedetect.getFace(imagePath, cascPath, 30)

if face is None:
    print('Found 0 or >1 faces')

cv2.imwrite(fileToSaveImage, face)
