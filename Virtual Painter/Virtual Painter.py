import cv2
import time
import HandTracking as htm
import numpy as np
import os



overlayList = []

brushThickness = 10
eraserThickness = 100
drawColor = (0, 0, 255)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

 
folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
for imPath in myList:  # reading all the images from the folder
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)  # inserting images one by one in the overlayList
header = overlayList[0]  # storing 1st image
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=1, maxHands=1)  # making object

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)  # using functions fo connecting landmarks
    lmList, bbox = detector.findPosition(img,
                                         draw=False)  # using function to find specific landmark position,draw false means no circles on landmarks

    if len(lmList) != 0:
        # print(lmList)
        x1, y1 = lmList[8][1], lmList[8][2]  # tip of index finger
        x2, y2 = lmList[12][1], lmList[12][2]  # tip of middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("Selection Mode")

            if y1 < 190:
                if 0 < x1 < 150:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 170 < x1 < 350:
                    header = overlayList[1]
                    drawColor = (0,165,255)
                elif 450 < x1 < 550:
                    header = overlayList[2]
                    drawColor = (0, 255, 255)
                elif 580 < x1 < 650:
                    header = overlayList[3]
                    drawColor = (0, 255, 0)
                elif 850 < x1 < 900:
                    header = overlayList[4]
                    drawColor = (255, 196, 9)
                elif 950 < x1 < 1100:
                    header = overlayList[5]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor,
                          cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # drawing mode is represented as circle
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1


            # eraser
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor,
                         brushThickness)  # gonna draw lines from previous coodinates to new positions
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1  # giving values to xp,yp everytime

        # merging two windows into one imgcanvas and img

    # 1 converting img to gray
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    # 2 converting into binary image and thn inverting
    _, imgInv = cv2.threshold(imgGray, 50, 255,
                              cv2.THRESH_BINARY_INV)  # on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask

    imgInv = cv2.cvtColor(imgInv,
                          cv2.COLOR_GRAY2BGR)  # converting again to gray bcoz we have to add in a RGB image i.e img

    # add original img with imgInv ,by doing this we get our drawing only in black color
    img = cv2.bitwise_and(img, imgInv)

    # add img and imgcanvas,by doing this we get colors on img
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    img[0:176, 0:1250] = header  # on our frame we are setting our JPG image acc to H,W of jpg images

    cv2.imshow("Image", img)
    cv2.imshow(("canvas"),imgCanvas)
    # cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)