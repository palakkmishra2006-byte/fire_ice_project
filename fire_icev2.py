import cv2
import mediapipe as mp
import numpy as np
import random

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# particles list
particles = []

def create_particle(x, y, mode):
    return {
        "x": x,
        "y": y,
        "vx": random.randint(-2, 2),
        "vy": random.randint(-5, -1),
        "life": random.randint(20, 40),
        "mode": mode
    }

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # 👆 index finger tip
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # ✨ generate particles
            for _ in range(3):
                mode = random.choice(["fire", "ice"])
                particles.append(create_particle(x, y, mode))

    # 🎨 update particles
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["mode"] == "fire":
            color = (0, 0, 255)   # red fire
        else:
            color = (255, 255, 255)  # ice white

        cv2.circle(img, (p["x"], p["y"]), 5, color, -1)

        if p["life"] <= 0:
            particles.remove(p)

    cv2.imshow("Fire Ice Hand Animation", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()