import sys
import facedetect

imagePath = sys.argv[1]
cascPath = sys.argv[2]

faces = facedetect.getFaces(imagePath, cascPath, 30)

for (x, y, w, h) in faces:
    print '(x: %d, y: %d, w: %d, h: %d)' % (x, y, w, h)
