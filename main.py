import numpy as np
import math
import cv2 as cv
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


with open("model\labels.txt", "r") as f:
    lines = f.readlines()
labels = [line.strip().split()[1] for line in lines]

cam = cv.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)
classifier = Classifier('model/hand_sign_detection_model.h5', 'Model/labels.txt')

offset = 20
imgSize = 300
counter = 0
last_time = time.time()
delay = 3 #time for changing character
max_time_on_char = 3 #time on 1 similar character
char_time = time.time()
sentence = ''
pre_char = ''

while True:
    ret, img = cam.read()
    imgOutput = img.copy()
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
                outputs, index = classifier.getPrediction(imgWhite)
                print(outputs, index)
                
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv.resize(imgCrop,(imgSize, hCal))
                imgShape = imgResize.shape
                hGap = (imgSize - hCal)//2
                imgWhite[hGap: hGap+hCal, :] = imgResize
                outputs, index = classifier.getPrediction(imgWhite)
                print(outputs, index)

            # Check if imgCrop dimensions are valid
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                character = labels[index]          
                if character == '_':
                    if sentence and sentence[-1] !='_':
                        sentence += '_'
                else: 
                    if character != pre_char and time.time()-last_time>delay:
                        sentence += character
                        pre_char = character
                        last_time = time.time()
                        char_time = time.time()
                    elif time.time() - char_time > max_time_on_char:
                        sentence += character
                        pre_char = character
                        char_time = time.time()

                cv.putText(imgOutput, sentence, (50,50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv.imshow("ImageWhite", imgWhite)

            else:
                print("Not Valid")
        else:
            print("Not Detection or Not Valid")    
    # Display the captured frame
    cv.imshow('Camera', imgOutput)

    key = cv.waitKey(1)
    if key == ord('r'):
        sentence = ''
        pre_char = ''
        last_time = time.time()
    elif key == ord('s'):
        with open ('text.txt', 'a') as f:
            sentence = sentence.replace('_', ' ')
            f.write(sentence+'\n')
        print('Saved', sentence)    
        sentence = ''
        pre_char = ''
    elif key == 27: #ESC Key to exit
     break


cam.release()
cv.destroyAllWindows()
