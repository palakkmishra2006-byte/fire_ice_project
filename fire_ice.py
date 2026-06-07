import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

mp_handedness = mp_hands.Hands()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for i, handLms in enumerate(result.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Hand type detection
            label = result.multi_handedness[i].classification[0].label

            if label == "Right":
                cv2.putText(img, "FIRE 🔥", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

            else:
                cv2.putText(img, "ICE ❄️", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)

    cv2.imshow("Fire Ice Model", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()