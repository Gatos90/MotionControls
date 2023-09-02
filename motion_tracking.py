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
from sklearn.svm import OneClassSVM
import pdb
import traceback

CSV_DIR = "models/cords"
MODEL_DIR = "models/models"
FEATURE_DIR = "models/features"


class Track_Object:
    def __init__(self):
        self.current_class_name = "DefaultClass"
        self.should_show_image = False
        self.exporting = False
        self.exportingLeftHand = False
        self.exportingRightHand = False
        self.exportingBodyPose = False
        self.should_train = False  # Flag to determine if training should occur
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=True,
        )
        self.models = self.load_all_models("models/models/")
        self.palm_start_time = None
        self.palm_end_time = None
        self.prev_y = [None] * 4
        self.velocity = [0] * 4
        self.jump_threshold = 0.08
        self.jump_velocity_threshold = 0.02
        self.saved_positions = None
        self.models = self.load_all_models("models/models/")
        start_time = time.time()
        end_time = time.time()
        self.loading_time = end_time - start_time

        # Load feature names here
        self.feature_names = {}
        for filename in os.listdir(FEATURE_DIR):
            if filename.endswith(".txt"):
                model_name = filename[
                    :-4
                ]  # removing the .txt extension to get the model name
                with open(os.path.join(FEATURE_DIR, filename), "r") as f:
                    self.feature_names[model_name] = [line.strip() for line in f]

    def get_csv_path_pose(self):
        """Returns the path to the Pose CSV for the current class."""
        return os.path.join(CSV_DIR, f"pose_coords_{self.current_class_name}.csv")

    def get_csv_path_right_hand(self):
        """Returns the path to the Hand CSV for the current class."""
        return os.path.join(CSV_DIR, f"right_hand_coords_{self.current_class_name}.csv")

    def get_csv_path_left_hand(self):
        """Returns the path to the Hand CSV for the current class."""
        return os.path.join(CSV_DIR, f"left_hand_coords_{self.current_class_name}.csv")

    def set_export_left_hand(self, value):
        self.exportingLeftHand = value

    def set_export_right_hand(self, value):
        self.exportingRightHand = value

    def set_export_body_pose(self, value):
        self.exportingBodyPose = value

    def get_model_path(self):
        """Returns the path to the model for the current class."""
        if self.exportingLeftHand:
            return os.path.join(MODEL_DIR, f"left_hand_{self.current_class_name}.pkl")
        if self.exportingRightHand:
            return os.path.join(MODEL_DIR, f"right_hand_{self.current_class_name}.pkl")
        if self.exportingBodyPose:
            return os.path.join(
                MODEL_DIR, f"body_language_{self.current_class_name}.pkl"
            )

    def get_feature_path(self):
        """Returns the path to the feature names file for the current class."""
        if self.exportingLeftHand:
            return os.path.join(
                FEATURE_DIR, f"left_hand_{self.current_class_name}.txt"
            )
        if self.exportingRightHand:
            return os.path.join(
                FEATURE_DIR, f"right_hand_{self.current_class_name}.txt"
            )
        if self.exportingBodyPose:
            return os.path.join(
                FEATURE_DIR, f"body_pose_{self.current_class_name}.txt"
            )

    def load_all_models(self, directory_path):
        model_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
        models = {}
        for model_file in model_files:
            with open(os.path.join(directory_path, model_file), "rb") as f:
                model_name = model_file[
                    :-4
                ]  # removing the .pkl extension to get the model's name
                models[model_name] = pickle.load(f)
        return models

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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

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

        brect = self.calc_bounding_rect(image, results.right_hand_landmarks)

        if brect is not None:
            self.draw_bounding_rect(True, image, brect)

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
        if results.pose_landmarks:
            num_coords_pose = len(results.pose_landmarks.landmark)
        else:
            num_coords_pose = 0
        landmarks_pose = ["class"]
        for val in range(1, num_coords_pose + 1):
            landmarks_pose += [
                "x{}".format(val),
                "y{}".format(val),
                "z{}".format(val),
                "v{}".format(val),
            ]

        # Get hand landmarks
        if results.right_hand_landmarks:
            num_coords_hand = len(results.right_hand_landmarks.landmark)
        else:
            num_coords_hand = 0
        landmarks_hand = ["class"]
        for val in range(1, num_coords_hand + 1):
            landmarks_hand += [
                "x{}".format(val),
                "y{}".format(val),
                "z{}".format(val),
                "v{}".format(val),
            ]

        if results.left_hand_landmarks:
            num_coords_hand = len(results.left_hand_landmarks.landmark)
        else:
            num_coords_hand = 0
        landmarks_hand = ["class"]
        for val in range(1, num_coords_hand + 1):
            landmarks_hand += [
                "x{}".format(val),
                "y{}".format(val),
                "z{}".format(val),
                "v{}".format(val),
            ]

        class_name = self.current_class_name

        # Log pose landmarks if checkbox is checked
        if self.exportingBodyPose and results.pose_landmarks:
            if not os.path.exists(self.get_csv_path_pose()):
                with open(self.get_csv_path_pose(), mode="w", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(landmarks_pose)
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
                row_pose = pose_row
                row_pose.insert(0, class_name)
                with open(self.get_csv_path_pose(), mode="a", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(row_pose)
            except Exception as e:
                print(f"Error exporting coordinates to CSV: {e}")

        # Log left hand landmarks if checkbox is checked
        if self.exportingLeftHand and results.left_hand_landmarks:
            if not os.path.exists(self.get_feature_path()):
                with open(self.get_feature_path(), mode="w", newline="") as f:
                    for val in ["class"] + landmarks_hand[1:]:
                        f.write(val + "\n")
            if not os.path.exists(self.get_csv_path_left_hand()):
                with open(self.get_csv_path_left_hand(), mode="w", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(landmarks_hand)
            try:
                hand = results.left_hand_landmarks.landmark
                hand_row = list(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in hand
                        ]
                    ).flatten()
                )
                row_hand = hand_row
                row_hand.insert(0, class_name)
                with open(self.get_csv_path_left_hand(), mode="a", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(row_hand)
            except Exception as e:
                print(f"Error exporting coordinates to CSV: {e}")

        # Log right hand landmarks if checkbox is checked
        if self.exportingRightHand and results.right_hand_landmarks:
            if not os.path.exists(self.get_feature_path()):
                with open(self.get_feature_path(), mode="w", newline="") as f:
                    for val in ["class"] + landmarks_hand[1:]:
                        f.write(val + "\n")
            if not os.path.exists(self.get_csv_path_right_hand()):
                with open(self.get_csv_path_right_hand(), mode="w", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(landmarks_hand)

            try:
                hand = results.right_hand_landmarks.landmark
                hand_row = list(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in hand
                        ]
                    ).flatten()
                )
                row_hand = hand_row
                row_hand.insert(0, class_name)
                with open(self.get_csv_path_right_hand(), mode="a", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(row_hand)
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
        if self.exportingRightHand:
            df = pd.read_csv(self.get_csv_path_right_hand())
        if self.exportingLeftHand:
            df = pd.read_csv(self.get_csv_path_left_hand())
        if self.exportingBodyPose:
            df = pd.read_csv(self.get_csv_path_pose())
        X = df.drop("class", axis=1)
        y = df["class"]

        # For one-class classification (anomaly detection)
        if len(y.unique()) == 1:
            # Using OneClassSVM for one-class classification
            model = make_pipeline(
                StandardScaler(), OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01)
            )
            model.fit(X)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=1234
            )
            # Convert back to DataFrame to retain column names
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

            # Train a Random Forest model
            pipeline_rf = make_pipeline(StandardScaler(), RandomForestClassifier())
            model = pipeline_rf.fit(X_train, y_train)

        with open(self.get_model_path(), "wb") as f:
            pickle.dump(model, f)

        end_time = time.time()  # Record the end time
        print(f"The train_model execution time : {end_time - start_time} seconds")

    def predict_model(self, image, results):
        #pdb.set_trace()
        for model_name, model in self.models.items():
            try:
                # Pose
                if "pose" in model_name and results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(
                        np.array(
                            [
                                [
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                    landmark.visibility,
                                ]
                                for landmark in pose
                            ]
                        ).flatten()
                    )
                    row = pose_row

                # RightHand
                elif "right_hand" in model_name and results.right_hand_landmarks:
                    hand = results.right_hand_landmarks.landmark
                    hand_row = list(
                        np.array(
                            [
                                [
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                    landmark.visibility,
                                ]
                                for landmark in hand
                            ]
                        ).flatten()
                    )
                    row = hand_row

                # LeftHand
                elif "left_hand" in model_name and results.left_hand_landmarks:
                    hand = results.left_hand_landmarks.landmark
                    hand_row = list(
                        np.array(
                            [
                                [
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                    landmark.visibility,
                                ]
                                for landmark in hand
                            ]
                        ).flatten()
                    )
                    row = hand_row

                else:
                    continue
                X = pd.DataFrame([row], columns=self.feature_names[model_name])
                
                y_pred = model.predict(X)
                if y_pred[0] == 1: 
                    print(f"A match for {model_name} has been found.")

            except Exception as e:
                print(f"Error in model '{model_name}': {e}")
                traceback.print_exc()      
        
            
                
    
    def set_class_name(self, class_name):
        """Sets the current class name for exporting."""
        if class_name:  # Ensure that the class name isn't empty
            self.current_class_name = class_name

    def calc_bounding_rect(self, image, landmarks):
        if landmarks is None:
            return None

        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv2.rectangle(
                image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2
            )
        return image
