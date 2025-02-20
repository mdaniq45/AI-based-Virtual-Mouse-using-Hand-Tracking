
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()
smooth_factor = 5  # Adjusts smooth movement

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Camera width
cap.set(4, 720)   # Camera height

prev_x, prev_y = 0, 0  # Previous cursor position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract key landmarks
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert coordinates to screen scale
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            screen_x = np.interp(index_x, [100, w - 100], [0, screen_w])
            screen_y = np.interp(index_y, [100, h - 100], [0, screen_h])

            # Smooth cursor movement
            cur_x = prev_x + (screen_x - prev_x) / smooth_factor
            cur_y = prev_y + (screen_y - prev_y) / smooth_factor

            # Move mouse cursor
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # Draw fingertip circle
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

            # Check finger distance for clicking
            thumb_dist = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y])
            )

            middle_dist = np.linalg.norm(
                np.array([middle_finger_tip.x, middle_finger_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            )

            # Left Click: Pinch index & thumb
            if thumb_dist < 0.05:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right Click: Pinch middle finger & thumb
            if middle_dist < 0.05:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show webcam feed
    cv2.imshow("AI Virtual Mouse", frame)
    
    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
