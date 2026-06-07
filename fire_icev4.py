import cv2
import mediapipe as mp
import random
import math
import numpy as np

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

# -------------------------
# MediaPipe
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

mp_draw = mp.solutions.drawing_utils

# -------------------------
# Data
# -------------------------
particles = []
trail = []

sx, sy = 0, 0
smooth_factor = 0.35

prev_x = 0
prev_y = 0

# -------------------------
# Window
# -------------------------
cv2.namedWindow("AR Magic", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(
    "AR Magic",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# -------------------------
# Helpers
# -------------------------
def smooth(prev, current):
    return int(prev * (1 - smooth_factor) + current * smooth_factor)

def create_particle(x, y, mode):

    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(1, 4)

    return {
        "x": x,
        "y": y,
        "vx": math.cos(angle) * speed,
        "vy": math.sin(angle) * speed - 1.5,
        "life": random.randint(25, 60),
        "size": random.randint(2, 5),
        "mode": mode
    }

def draw_glow(img, x, y, color, radius):

    overlay = img.copy()

    for r in range(radius * 2, radius, -5):

        alpha = max(0.03, r / (radius * 10))

        cv2.circle(
            overlay,
            (x, y),
            r,
            color,
            -1
        )

        cv2.addWeighted(
            overlay,
            alpha,
            img,
            1 - alpha,
            0,
            img
        )

def draw_crystal(img, x, y, size, color):

    cv2.circle(img, (x, y), size, color, -1)

    cv2.circle(
        img,
        (x - size // 3, y - size // 3),
        max(1, size // 3),
        (255, 255, 255),
        -1
    )

    cv2.line(img, (x-size, y), (x+size, y), (255,255,255), 1)
    cv2.line(img, (x, y-size), (x, y+size), (255,255,255), 1)

# -------------------------
# Main Loop
# -------------------------
while True:

    success, img = cap.read()

    if not success:
        continue

    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for i, handLms in enumerate(result.multi_hand_landmarks):

            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            # Index fingertip
            tip = handLms.landmark[8]

            x = int(tip.x * w)
            y = int(tip.y * h)

            sx = smooth(sx, x)
            sy = smooth(sy, y)

            # speed burst
            speed = math.sqrt(
                (sx - prev_x) ** 2 +
                (sy - prev_y) ** 2
            )

            prev_x = sx
            prev_y = sy

            label = result.multi_handedness[i].classification[0].label

            mode = "fire" if label == "Right" else "ice"

            # Trail
            trail.append((sx, sy, mode))

            if len(trail) > 40:
                trail.pop(0)

            # Normal particles
            for _ in range(3):
                particles.append(
                    create_particle(sx, sy, mode)
                )

            # Burst
            if speed > 20:

                for _ in range(15):
                    particles.append(
                        create_particle(sx, sy, mode)
                    )

            # Palm glow orb
            px = int(handLms.landmark[0].x * w)
            py = int(handLms.landmark[0].y * h)

            if mode == "fire":
                draw_glow(
                    img,
                    px,
                    py,
                    (0,180,255),
                    25
                )
            else:
                draw_glow(
                    img,
                    px,
                    py,
                    (255,255,255),
                    25
                )

    # -------------------------
    # Trail render
    # -------------------------
    for i, item in enumerate(trail):

        tx, ty, mode = item

        fade = i / max(len(trail), 1)

        radius = max(1, int(8 * fade))

        if mode == "fire":
            color = (0, int(180*fade), 255)
        else:
            color = (255,255,255)

        cv2.circle(
            img,
            (tx, ty),
            radius,
            color,
            -1
        )

    # -------------------------
    # Particle render
    # -------------------------
    for p in particles[:]:

        p["vy"] += 0.15

        p["x"] += p["vx"]
        p["y"] += p["vy"]

        p["life"] -= 1

        fade = p["life"] / 60

        if p["mode"] == "fire":

            palette = [
                (0,120,255),
                (0,180,255),
                (0,220,255),
                (50,255,255)
            ]

        else:

            palette = [
                (255,255,255),
                (255,240,220),
                (255,220,180)
            ]

        color = random.choice(palette)

        brightness = random.uniform(0.8, 1.3)

        color = (
            min(255, int(color[0]*brightness)),
            min(255, int(color[1]*brightness)),
            min(255, int(color[2]*brightness))
        )

        size = max(
            1,
            int(p["size"] * fade)
        )

        draw_crystal(
            img,
            int(p["x"]),
            int(p["y"]),
            size,
            color
        )

        if p["life"] <= 0:
            particles.remove(p)

    cv2.imshow("AR Magic", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()