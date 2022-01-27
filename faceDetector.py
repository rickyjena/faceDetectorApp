import cv2

# This is pre-trained data that is being loaded from opencv
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Take in webcam video with cv2
webcam = cv2.VideoCapture(0)

# Loop through frames of video
while True:

    # Take in current frame
    successfullyRead, capturedFrame = webcam.read()

    # We will convert image to grayscale for detector requirements
    grayScaleImg = cv2.cvtColor(capturedFrame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faceCoords = trainedFaceData.detectMultiScale(grayScaleImg)

    for x in faceCoords:
        (startX, startY, width, height) = x
        startPoint = (startX, startY)
        endPoint = (startX + width, startY + height)

        # Set box color
        boxColor = (0, 255, 0)

        # Box side width
        boxWidth = 2

        # Draw rectangles around the faces using coords
        cv2.rectangle(capturedFrame, startPoint, endPoint, boxColor, boxWidth)
    
    # Show each frame from webcam
    cv2.imshow('Face Detector', capturedFrame)
    
    # This is used to choose how long each frame is shown for in milliseconds
    key = cv2.waitKey(1)

    # Pressin lowercase b will exit the loop and end the program
    if key == 98:
        break

# Releases the the webcam
webcam.release()


"""
######################This is to detect faces in image
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
"""