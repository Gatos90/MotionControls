# motion_tracker.py

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class MotionTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.prev_y = [None] * 4
        self.velocity = [0] * 4
        self.jump_threshold = 0.08
        self.jump_velocity_threshold = 0.02

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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
                    if self.prev_y[i] is not None:
                        self.velocity[i] = y - self.prev_y[i]

                if None not in self.prev_y:
                    shoulder_change = abs(self.prev_y[0] - cur_y[0]) + abs(self.prev_y[1] - cur_y[1])
                    hip_change = abs(self.prev_y[2] - cur_y[2]) + abs(self.prev_y[3] - cur_y[3])

                    shoulder_velocity = self.velocity[0] + self.velocity[1]
                    hip_velocity = self.velocity[2] + self.velocity[3]

                    if (
                        shoulder_change > self.jump_threshold
                        and shoulder_velocity > self.jump_velocity_threshold
                    ) and (
                        hip_change > self.jump_threshold
                        and hip_velocity > self.jump_velocity_threshold
                    ):
                        print(
                            f"Jump detected: Shoulder Change={shoulder_change}, Hip Change={hip_change}"
                        )

                self.prev_y = cur_y.copy()

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

        return image

    def release(self):
        self.cap.release()
