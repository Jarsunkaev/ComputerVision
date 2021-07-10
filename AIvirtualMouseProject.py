from typing import List, Callable

import cv2
import numpy as np
import pyautogui

import HandTrackingModule as htm
import time

###############################
wCam, hCam = 640, 480
frameR = 100  # Frame reduction
smoothening = 4
###############################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()
# print(wScr, hScr)

while True:
    # 1. find hand lmd
    success, img = cap.read()
    img = detector.findhands(img)
    lmList, bbox = detector.findPosition(img)
    # print(lmList)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)

        # tip of index and middle fingers
        # check which fingers are up
        fingers: Callable[[], list[int]] = detector.fingersUp
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255, 2))

        if fingers[1] == 1 and fingers[2] == 0:
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening


            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if 1 != fingers[1] or fingers[2] != 1:
            continue
        length, img, lineInfo = detector.findDistance(8, 12, img)

        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255, 0), cv2.FILLED)
            pyautogui.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
