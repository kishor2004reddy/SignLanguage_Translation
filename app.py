import cv2 as cv

cv.namedWindow("Sign Language Translator", cv.WINDOW_NORMAL)

# realtime_asl_translator.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
from collections import deque
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

with open("asl_labels.json") as f:
    labels = json.load(f)
num_classes = len(labels)

# Load model + labels
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
conv_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = True
for layer in conv_base.layers[:-30]:
    layer.trainable = False
x = conv_base(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

model = Model(inputs, outputs)

model.load_weights("model.h5")



# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

IMG_SIZE = 224
sentence = []
prediction_buffer = deque(maxlen=15)  # to smooth predictions
last_word = ""

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural webcam view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            ymin, ymax = int(min(y_coords)) - 20, int(max(y_coords)) + 20
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            # Crop & preprocess
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size != 0:
                img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                # Predict
                pred = model.predict(img, verbose=0)
                pred_class = labels[str(np.argmax(pred))]
                prediction_buffer.append(pred_class)

                # Stable prediction
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                    if most_common != last_word:
                        sentence.append(most_common)
                        last_word = most_common

            # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display sentence
    cv2.putText(frame, " ".join(sentence), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
