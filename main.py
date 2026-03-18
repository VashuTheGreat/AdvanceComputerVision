import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, n_hands=2, dConf=0.5, tConf=0.5):
        self.mode = mode
        self.n_hands = n_hands
        self.dConf = dConf
        self.tConf = tConf

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.n_hands,
            min_detection_confidence=self.dConf,
            min_tracking_confidence=self.tConf
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            return self.results.multi_hand_landmarks
        
        return []

    def drawHands(self, img, hands, draw=True):
        if draw:
            for hand in hands:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img


if __name__ == "__main__":
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