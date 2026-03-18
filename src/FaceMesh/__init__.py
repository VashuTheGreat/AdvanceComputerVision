import cv2
import mediapipe as mp
import time

mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True
)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("public/video.mp4")

pTime = 0

while True:
    success, img = cap.read()

    if not success:
        break 

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                img,
                faceLms,
                mpFaceMesh.FACEMESH_TESSELATION
            )

    cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()