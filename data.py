import numpy as np
import math
import cv2 as cv
import time
from cvzone.HandTrackingModule import HandDetector

folder = 'assets/backspace'

cam = cv.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

while True:
    ret, img = cam.read()
    hands, img = hand_detector.findHands(img) 
    height, width, _ = img.shape
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        if w > 0 and h > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

            # Crop image in valid domain
            imgCrop = img[y:y + h, x:x + w]
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv.resize(imgCrop,(wCal, imgSize))
                imgShape = imgResize.shape
                wGap = (imgSize - wCal)//2
                imgWhite[:, wGap: wGap+wCal] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv.resize(imgCrop,(imgSize, hCal))
                imgShape = imgResize.shape
                hGap = (imgSize - hCal)//2
                imgWhite[hGap: hGap+hCal, :] = imgResize

            # Check if imgCrop dimensions are valid
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                cv.imshow('ImageCrop', imgCrop)
                cv.imshow("ImageWhite", imgWhite)
            else:
                print("Not Valid")
        else:
            print("Not Detection or Not Valid")    
    # Display the captured frame
    cv.imshow('Camera', img)

    key = cv.waitKey(1)
    if key == 27: #ESC Key to exit
     break
    
    if key == ord('s'):
        counter +=1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f'{counter}')


cam.release()
cv.destroyAllWindows()

