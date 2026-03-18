import cv2
import mediapipe as mp

import time


mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection()

mpDraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture("public/video.mp4")

pTime=0
while True:
    success,img=cap.read()

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results=face.process(imgRGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)
            print(detection.location_data.relative_bounding_box)
    cv2.putText(img,f"Fps: {int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)

    cv2.imshow("Image",img)

    cv2.waitKey(1)