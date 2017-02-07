import cv2

def getFace(imagePath, cascPath, searchSize):
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (searchSize, searchSize),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print 'Found: ', len(faces), ' faces'
    for (x, y, w, h) in faces: print '(x: %d, y: %d, w: %d, h: %d)' % (x, y, w, h)
    if len(faces) != 1:
        return None
    else:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        crop_img = image[y: y + h, x: x + w]
        return crop_img
