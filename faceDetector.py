import cv2

# This is pre-trained data that is being loaded from opencv
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Image sample to detect face
img = cv2.imread('243315131_6485777301463164_7029905873815792864_n.jpg')

# We will convert image to grayscale for detector requirements
grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faceCoords = trainedFaceData.detectMultiScale(grayScaleImg)

# Print Coordinates ([upperleft coord, (width,height)])
print(faceCoords)

startPoint = (faceCoords[0][0],faceCoords[0][1])
endPoint = (faceCoords[0][0]+faceCoords[0][2],faceCoords[0][1]+faceCoords[0][3])
boxColor = (0, 255, 0)

# Draw rectangles around the faces using coords
cv2.rectangle(img, startPoint, endPoint, boxColor, 2)

# Show sample image
cv2.imshow('Face Detector', img)
cv2.waitKey()

print("my Face")