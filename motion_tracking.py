import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

prev_y = [None] * 4
velocity = [0] * 4
jump_threshold = 0.08  # Set threshold for jump detection
jump_velocity_threshold = 0.02  # Set threshold for velocity to consider as a jump

window = tk.Tk()
canvas = tk.Canvas(window, width = cap.get(3), height = cap.get(4))
canvas.pack()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()

        # Check if frame was captured successfully
        if not ret:
            print("Frame not captured successfully")
            break

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        if results.pose_landmarks:
            cur_y = []
            landmarks_to_check = [
                mp_holistic.PoseLandmark.LEFT_SHOULDER,
                mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                mp_holistic.PoseLandmark.LEFT_HIP,
                mp_holistic.PoseLandmark.RIGHT_HIP,
            ]

            for i, landmark in enumerate(landmarks_to_check):
                y = results.pose_landmarks.landmark[landmark].y
                cur_y.append(y)
                if prev_y[i] is not None:
                    velocity[i] = y - prev_y[i]

            if None not in prev_y:
                shoulder_change = abs(prev_y[0] - cur_y[0]) + abs(prev_y[1] - cur_y[1])
                hip_change = abs(prev_y[2] - cur_y[2]) + abs(prev_y[3] - cur_y[3])

                shoulder_velocity = velocity[0] + velocity[1]
                hip_velocity = velocity[2] + velocity[3]

                if (
                    shoulder_change > jump_threshold
                    and shoulder_velocity > jump_velocity_threshold
                ) and (
                    hip_change > jump_threshold
                    and hip_velocity > jump_velocity_threshold
                ):
                    print(
                        f"Jump detected: Shoulder Change={shoulder_change}, Hip Change={hip_change}"
                    )

            prev_y = cur_y.copy()
        else:
            print("Pose landmarks not found")

        image.flags.writeable = True

        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

        photo = ImageTk.PhotoImage(image = Image.fromarray(image))

        canvas.create_image(0, 0, image = photo, anchor = tk.NW)

        window.update_idletasks()
        window.update()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            print("Exit command received, stopping...")
            break

cap.release()
cv2.destroyAllWindows()
print("End of program")
