import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            ymin, ymax = int(min(y_coords)) - 20, int(max(y_coords)) + 20
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    if cv.waitKey(1) & 0xFF == ord('s'):
        hand_img = frame[ymin:ymax, xmin:xmax]
        if hand_img.size != 0:
            cv.imwrite("captured_hand.jpg", hand_img)
            

    cv.imshow("Sign Language Translator", frame)
    

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.waitKey(0)
video.release()