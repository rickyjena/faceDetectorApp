import cv2

# This is pre-trained data that is being loaded from opencv
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Image sample to detect face
img = cv2.imread('243315131_6485777301463164_7029905873815792864_n.jpg')

# Show sample image
cv2.imshow('Face Detector', img)
cv2.waitKey()

print("my Face")