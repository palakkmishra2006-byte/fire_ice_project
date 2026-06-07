import cv2
import mediapipe as mp
import random
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

particles = []

# ✨ Create particle
def create_particle(x, y, mode):
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(0.5, 2)

    return {
        "x": x,
        "y": y,
        "vx": math.cos(angle) * speed,
        "vy": math.sin(angle) * speed - 1,
        "life": random.randint(25, 50),
        "mode": mode,
        "size": random.randint(3, 6)
    }

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for i, handLms in enumerate(result.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # 👆 Index finger tip
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # ✋ Hand type (Left / Right)
            label = result.multi_handedness[i].classification[0].label

            # 🔥 Right hand = Fire
            if label == "Right":
                mode = "fire"
            # ❄️ Left hand = Ice
            else:
                mode = "ice"

            # ✨ particle generation (smooth trail, not too chaotic)
            for _ in range(2):
                particles.append(create_particle(x, y, mode))

    # 🎨 draw particles
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        alpha = max(p["life"] / 50, 0)

        if p["mode"] == "fire":
            # 🔥 orange crystal glow
            color = (0, int(140 * alpha), int(255 * alpha))  # orange-gold
        else:
            # ❄️ white-blue crystal
            color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))

        # ✨ soft size change (fade effect)
        size = max(int(p["size"] * alpha), 1)

        cv2.circle(img, (int(p["x"]), int(p["y"])), size, color, -1)

        # remove dead particles
        if p["life"] <= 0:
            particles.remove(p)

    cv2.imshow("Fire Ice Crystal AR", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()