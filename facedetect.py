import cv2

def getFaces(imagePath, cascPath, searchSize):
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
    return faces
