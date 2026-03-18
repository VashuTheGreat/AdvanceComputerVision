import cv2
import mediapipe as mp

from HandDetectorModule import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        hands = detector.findHands(img)

        if hands:
            img = detector(img, hands)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()