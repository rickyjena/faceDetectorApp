import cv2

# This is pre-trained data that is being loaded from opencv
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Image sample to detect face
img = cv2.imread('4faceGroup.jpg')

# We will convert image to grayscale for detector requirements
grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faceCoords = trainedFaceData.detectMultiScale(grayScaleImg)

# Print Coordinates ([upperleft coord, (width,height)])
# print(faceCoords, "\n")

for x in faceCoords:
    (startX, startY, width, height) = x
    startPoint = (startX, startY)
    endPoint = (startX + width, startY + height)
    # print("Starting Point:", startPoint,"\n"
    # "End point:",endPoint,"\n")
    # Set box color
    boxColor = (0, 255, 0)

    # Box side width
    boxWidth = 2

    # Draw rectangles around the faces using coords
    cv2.rectangle(img, startPoint, endPoint, boxColor, boxWidth)

# Show sample image
cv2.imshow('Face Detector', img)
cv2.waitKey()

print("Face has been detected in image")