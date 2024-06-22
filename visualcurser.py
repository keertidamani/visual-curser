import cv2
import mediapipe as mp
import pyautogui
import time

FRAME_RATE = 30
CLICK_THRESHOLD = 0.004  # Threshold for detecting blink
SMOOTHING_FACTOR = 0.1
FRAME_INTERVAL = 1 / FRAME_RATE  # Interval between frames

# Initialize Camera and Hand & Face Mesh
cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

prev_screen_x, prev_screen_y = screen_w // 2, screen_h // 2
last_click_time = 0  # Initialize last click time
eye_control_active = True  # Toggle for eye control

# Scaling factors to increase sensitivity and reach
SCALING_FACTOR_X = 1.5
SCALING_FACTOR_Y = 1.5


def process_frame(frame):
    """Process the frame to detect facial landmarks, hand landmarks, and control mouse."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_hands = mp_hands.process(rgb_frame)
    output_face_mesh = mp_face_mesh.process(rgb_frame)
    return output_hands.multi_hand_landmarks, output_face_mesh.multi_face_landmarks

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on the frame."""
    if hand_landmarks:
        for landmark in hand_landmarks[0].landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

def move_cursor(landmark):
    """Move the mouse cursor to the position of the given landmark."""
    global prev_screen_x, prev_screen_y
    screen_x = int((landmark.x - 0.5) * SCALING_FACTOR_X * screen_w + screen_w / 2)
    screen_y = int((landmark.y - 0.5) * SCALING_FACTOR_Y * screen_h + screen_h / 2)
    # Smoothing
    screen_x = int(prev_screen_x * (1 - SMOOTHING_FACTOR) + screen_x * SMOOTHING_FACTOR)
    screen_y = int(prev_screen_y * (1 - SMOOTHING_FACTOR) + screen_y * SMOOTHING_FACTOR)
    pyautogui.moveTo(screen_x, screen_y)
    prev_screen_x, prev_screen_y = screen_x, screen_y

def detect_blink(landmarks):
    """Detect blink by comparing vertical distance between eye landmarks."""
    left_eye = [landmarks[145], landmarks[159]]
    right_eye = [landmarks[374], landmarks[386]]
    left_blink = (left_eye[0].y - left_eye[1].y) < CLICK_THRESHOLD
    right_blink = (right_eye[0].y - right_eye[1].y) < CLICK_THRESHOLD
    return left_blink or right_blink

def detect_hand_scroll(hand_landmarks):
    """Detect hand scroll gesture (e.g., thumb and index finger pinch)."""
    if hand_landmarks:
        thumb = hand_landmarks[0].landmark[4]  # Thumb tip landmark
        index_finger = hand_landmarks[0].landmark[8]  # Index finger tip landmark
        thumb_x, thumb_y = thumb.x * screen_w, thumb.y * screen_h
        index_x, index_y = index_finger.x * screen_w, index_finger.y * screen_h
        distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
        if distance < 100:  # Adjust threshold as needed
            return 'up', 8  # Scroll up faster (adjust scroll amount)
        elif distance > 200:  # Adjust threshold as needed
            return 'down', 8  # Scroll down faster (adjust scroll amount)
    return None, 1  # Default scroll amount

def main():
    global last_click_time, eye_control_active
    while True:
        start_time = time.time()
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        frame = cv2.flip(frame, 1)
        hand_landmarks, face_landmarks = process_frame(frame)

        if eye_control_active and face_landmarks:
            landmarks = face_landmarks[0].landmark
            draw_hand_landmarks(frame, hand_landmarks)
            move_cursor(landmarks[475])

            hand_scroll_direction, scroll_amount = detect_hand_scroll(hand_landmarks)
            if hand_scroll_direction == 'up':
                pyautogui.scroll(1 * scroll_amount)
            elif hand_scroll_direction == 'down':
                pyautogui.scroll(-1 * scroll_amount)

            if detect_blink(landmarks):
                current_time = time.time()
                if current_time - last_click_time > 1:
                    pyautogui.click()
                    last_click_time = current_time

        cv2.imshow('Eye Controlled Mouse', frame)

        # Toggle eye control on pressing 'space'
        if cv2.waitKey(1) & 0xFF == ord(' '):
            eye_control_active = not eye_control_active

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Control frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(0, FRAME_INTERVAL - elapsed_time)
        time.sleep(sleep_time)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
