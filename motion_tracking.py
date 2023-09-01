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
from sklearn.metrics import accuracy_score
import pickle

CSV_PATH = "models/cords/coords.csv"
MODEL_PATH = "models/models/body_language.pkl"


class Track_Object:
    def __init__(self):
        self.current_class_name = "DefaultClass"
        self.should_show_image = False
        self.exporting = False
        self.should_train = False  # Flag to determine if training should occur
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
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
        self.saved_positions = None
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                start_time = time.time()
                self.model = pickle.load(f)
                print(self.model)
                end_time = time.time()
                self.loading_time = end_time - start_time
                print(f"Model loading time: {self.loading_time} seconds")
            # Load feature names here
            with open("models/features/feature_names.txt", "r") as f:
                self.feature_names = [line.strip() for line in f]

    def draw_landmarks(self, image, landmarks, connections, color1, color2):
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            connections,
            self.mp_drawing.DrawingSpec(color=color1, thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=color2, thickness=2, circle_radius=2),
        )

    def process_image(self, frame):
        start_time = time.time()
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

        if results.face_landmarks:
            self.draw_landmarks(
                image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                (80, 110, 10),
                (80, 256, 121),
            )
        self.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            (80, 22, 10),
            (80, 44, 121),
        )
        self.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            (121, 22, 76),
            (121, 44, 250),
        )
        self.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            (245, 117, 66),
            (245, 66, 230),
        )

        if self.exporting:
            self.export_coordinates_to_csv(results)

        if self.should_train:
            try:
                self.train_model()
                self.should_train = False
            except Exception as e:
                print(f"Error during model training: {e}")

        try:
            self.predict_model(image, results)
        except Exception as e:
            print(f"Error during model prediction: {e}")

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.should_show_image:
            cv2.imshow("Output Window", image)
            cv2.waitKey(20)
        end_time = time.time()

    def export_coordinates_to_csv(self, results):
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

        if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
            with open(CSV_PATH, mode="w", newline="") as f:
                csv_writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(landmarks)

        class_name = self.current_class_name
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in pose
                    ]
                ).flatten()
            )
            face = results.face_landmarks.landmark
            face_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in face
                    ]
                ).flatten()
            )
            row = pose_row + face_row
            row.insert(0, class_name)
            with open(CSV_PATH, mode="a", newline="") as f:
                csv_writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(row)
        except Exception as e:
            print(f"Error exporting coordinates to CSV: {e}")

    def __del__(self):
        self.holistic.close()

    def start_export(self):
        self.exporting = True

    def stop_export(self):
        self.exporting = False

    def start_training(self):
        self.should_train = True

    def should_show_video(self):
        self.should_show_image = True

    def train_model(self):
        start_time = time.time()  # Record the start time
        df = pd.read_csv(CSV_PATH)
        X = df.drop("class", axis=1)
        y = df["class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1234
        )

        # Convert back to DataFrame to retain column names
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        pipelines = {
            "lr": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000)
            ),  # Increase the max_iter value
            "rc": make_pipeline(StandardScaler(), RidgeClassifier()),
            "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
            "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }

        fit_models = {}
        for algo, pipeline in pipelines.items():
            self.model = pipeline.fit(X_train, y_train)
            fit_models[algo] = self.model

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(fit_models["rf"], f)
            end_time = time.time()  # Record the end time
        with open("feature_names.txt", "w") as f:
            for col in X_train.columns:
                f.write(col + "\n")
            print(f"The train_model execution time : {end_time - start_time} seconds")

    def predict_model(self, image, results):
        if self.model is None:
            return
        pose = results.pose_landmarks.landmark
        pose_row = list(
            np.array(
                [
                    [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in pose
                ]
            ).flatten()
        )
        face = results.face_landmarks.landmark
        face_row = list(
            np.array(
                [
                    [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in face
                ]
            ).flatten()
        )

        row = pose_row + face_row
        X = pd.DataFrame([row], columns=self.feature_names)
        body_language_class = self.model.predict(X)[0]
        body_language_prob = self.model.predict_proba(X)[0]
        class_index = list(self.model.classes_).index(body_language_class)
        prob_of_predicted_class = body_language_prob[class_index]
        if prob_of_predicted_class * 100 > 70:
            print(
                f"Predicted Class: {body_language_class}, Probability: {prob_of_predicted_class*100}%"
            )
            coords = tuple(
                np.multiply(
                    np.array(
                        (
                            results.pose_landmarks.landmark[
                                self.mp_holistic.PoseLandmark.LEFT_EAR
                            ].x,
                            results.pose_landmarks.landmark[
                                self.mp_holistic.PoseLandmark.LEFT_EAR
                            ].y,
                        )
                    ),
                    [640, 480],
                ).astype(int)
            )
            cv2.rectangle(
                image,
                (coords[0], coords[1] + 5),
                (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                (245, 117, 16),
                -1,
            )
            cv2.putText(
                image,
                body_language_class,
                coords,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Get status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(
                image,
                "CLASS",
                (95, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                body_language_class.split(" ")[0],
                (90, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display Probability
            cv2.putText(
                image,
                "PROB",
                (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def set_class_name(self, class_name):
        """Sets the current class name for exporting."""
        if class_name:  # Ensure that the class name isn't empty
            self.current_class_name = class_name
