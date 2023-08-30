# motion_tracking.py
import time
import cv2
import mediapipe as mp


class Track_Object:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic  # Changed to lowercase 'h'
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.palm_start_time = None
        self.palm_end_time = None
        self.prev_y = [None] * 4
        self.velocity = [0] * 4
        self.jump_threshold = 0.08
        self.jump_velocity_threshold = 0.02
        self.saved_positions = None  # New variable for saved positions

    def save_pose(self):  # Function to save current pose
        self.saved_positions = self.prev_y.copy()

    def check_saved_positions(self, cur_y):  # Checking function
        if self.saved_positions:
            for prev, cur in zip(self.saved_positions, cur_y):
                if abs(prev - cur) > 0.01:
                    return False
            print("Hello")
            return True
        return False

    def process_image(self, frame):
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.holistic.process(image)  # Use self.holistic instance

        if results.pose_landmarks:
            cur_y = []
            landmarks_to_check = [
                self.mp_holistic.PoseLandmark.LEFT_SHOULDER,
                self.mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                self.mp_holistic.PoseLandmark.LEFT_HIP,
                self.mp_holistic.PoseLandmark.RIGHT_HIP,
            ]

            for i, landmark in enumerate(landmarks_to_check):
                y = results.pose_landmarks.landmark[landmark].y
                cur_y.append(y)

                if self.prev_y[i] is not None:
                    self.velocity[i] = y - self.prev_y[i]

        if None not in self.prev_y:
            shoulder_change = abs(self.prev_y[0] - cur_y[0]) + abs(
                self.prev_y[1] - cur_y[1]
            )
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

            self.check_saved_positions(
                cur_y
            )  # Call the saved position checking function

        self.prev_y = cur_y.copy()
        image.flags.writeable = True
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
        )

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def check_palm_duration(self):
        if self.palm_end_time - self.palm_start_time >= 3:
            print("Hello")

    def __del__(self):
        self.holistic.close()
