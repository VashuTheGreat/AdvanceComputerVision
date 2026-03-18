import mediapipe as mp
import time
import cv2

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw=mp.solutions.drawing_utils
cap = cv2.VideoCapture("public/video.mp4")

pTime = 0

while True:
    success, img = cap.read()

    if not success:
        break 

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()