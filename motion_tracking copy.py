# motion_tracking.py

# Required libraries are imported.
import time
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class Track_Object:
    def __init__(self):
        self.exporting = False
        # Initializing the drawing utilities to visualize the tracking.
        self.mp_drawing = mp.solutions.drawing_utils
        # Initializing the holistic model for full body landmark detection.
        self.mp_holistic = mp.solutions.holistic
        # Drawing styles for better visualization
        self.mp_drawing_styles = mp.solutions.drawing_styles
        # Setting up the Holistic model with specified min_detection and tracking confidence.
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        # Variables used for handling palm-related motion tracking.
        self.palm_start_time = None
        self.palm_end_time = None

        # Variables used to store previous y-coordinate positions of the four points of interest.
        self.prev_y = [None] * 4
        # Stores the velocity (rate of change of position) of each point. Initialized to zero.
        self.velocity = [0] * 4
        # Threshold values to determine if a jump has occurred based on positional changes and velocities.
        self.jump_threshold = 0.08
        self.jump_velocity_threshold = 0.02
        # Variable to save the positions.
        self.saved_positions = None

    def save_pose(self):
        # Function to save the current pose by copying the current y-coordinates list.
        self.saved_positions = self.prev_y.copy()

    def check_saved_positions(self, cur_y):
        # Function thats checks if the current positions are the same as the saved ones.
        # If any coordinate difference is greater than a certain threshold, returns False otherwise True.
        if self.saved_positions:
            for prev, cur in zip(self.saved_positions, cur_y):
                if abs(prev - cur) > 0.01:
                    return False
            print("Hello")
            return True
        return False

    def process_image(self, frame):
        # Start by initializing cur_y as an empty list.
        cur_y = []

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)

        # Extracting landmark data if available.
        if results.pose_landmarks:
            landmarks_to_check = [
                self.mp_holistic.PoseLandmark.LEFT_SHOULDER,
                self.mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                self.mp_holistic.PoseLandmark.LEFT_HIP,
                self.mp_holistic.PoseLandmark.RIGHT_HIP,
            ]

            for i, landmark in enumerate(landmarks_to_check):
                y = results.pose_landmarks.landmark[landmark].y
                cur_y.append(y)  # add this line here

                if self.prev_y[i] is not None:
                    self.velocity[i] = y - self.prev_y[i]

        # Calculating total shoulder/hip changes and velocities.
        if None not in self.prev_y and len(cur_y) >= 4:
            shoulder_change = abs(self.prev_y[0] - cur_y[0]) + abs(
                self.prev_y[1] - cur_y[1]
            )
            hip_change = abs(self.prev_y[2] - cur_y[2]) + abs(self.prev_y[3] - cur_y[3])
            shoulder_velocity = self.velocity[0] + self.velocity[1]
            hip_velocity = self.velocity[2] + self.velocity[3]

            # Detecting jump action.
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

            # Checking whether current positions match the saved ones.
            self.check_saved_positions(cur_y)

        # Updating the previous y coordinates only if cur_y has values
        if cur_y:
            self.prev_y = cur_y.copy()

        # Making the image writeable again and then visualizing all the landmark detections.
        image.flags.writeable = True

        # 1. Face
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(
                color=(80, 110, 10), thickness=1, circle_radius=1
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 256, 121), thickness=1, circle_radius=1
            ),
        )

        # 2. Right hand
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(80, 22, 10), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 44, 121), thickness=2, circle_radius=2
            ),
        )

        # 3. Left Hand
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(121, 44, 250), thickness=2, circle_radius=2
            ),
        )

        # 4. Pose Detections
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

        num_coords = len(results.pose_landmarks.landmark) + len(
            results.face_landmarks.landmark
        )

        landmarks = ["class"]
        for val in range(1, num_coords + 1):
            landmarks += [
                "x{}".format(val),
                "y{}".format(val),
                "z{}".format(val),
                "v{}".format(val),
            ]

        # Export coordinates
        if self.exporting:
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in pose
                        ]
                    ).flatten()
                )

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in face
                        ]
                    ).flatten()
                )

                # Concate rows
                row = pose_row + face_row

                class_name = "Happy"
                # Append class name
                row.insert(0, class_name)

                # Export to CSV
                with open("coords.csv", mode="a", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(row)

            except:
                pass

        # Convert image from RGB to BGR format, because OpenCV uses BGR (not RGB).
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Return the processed image.
        return image

    # Function to check palm open duration (calculated by difference between end time and start time).
    def check_palm_duration(self):
        # Check if the duration is greater or equal to 3 seconds.
        if self.palm_end_time - self.palm_start_time >= 3:
            # If duration is more than 3 seconds, print "Hello".
            print("Hello")

    # This special method is used for cleaning up resources.
    def __del__(self):
        # When an object of class Track_Object is deleted, close the holistic model.
        self.holistic.close()

    def start_export(self):
        self.exporting = True

    def stop_export(self):
        self.exporting = False
