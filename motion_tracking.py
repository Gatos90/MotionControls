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
        """Returns the path to the feature names file based on the current export category."""
        if self.exportingLeftHand:
            return os.path.join(FEATURE_DIR, "left_hand.txt")
        if self.exportingRightHand:
            return os.path.join(FEATURE_DIR, "right_hand.txt")
        if self.exportingBodyPose:
            return os.path.join(FEATURE_DIR, "body_pose.txt")

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
                self.train_data()
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
            print(f"Right hand landmarks data: {results.right_hand_landmarks}")
            num_coords_hand_right = len(results.right_hand_landmarks.landmark)
        else:
            num_coords_hand_right = 0
        landmarks_right_hand = ["class"]
        for val in range(1, num_coords_hand_right + 1):
            landmarks_right_hand += [
                "x{}".format(val),
                "y{}".format(val),
                "z{}".format(val),
                "v{}".format(val),
            ]

        if results.left_hand_landmarks:
            num_coords_hand_left = len(results.left_hand_landmarks.landmark)
        else:
            num_coords_hand_left = 0
        landmarks_left_hand = ["class"]
        for val in range(1, num_coords_hand_left + 1):
            landmarks_left_hand += [
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
            feature_path = self.get_feature_path()
            if not os.path.exists(feature_path):
                with open(feature_path, mode="w", newline="") as f:
                    for val in ["class"] + landmarks_left_hand[1:]:
                        f.write(val + "\n")
            if not os.path.exists(self.get_csv_path_left_hand()):
                with open(self.get_csv_path_left_hand(), mode="w", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(landmarks_left_hand)

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
            feature_path = self.get_feature_path()
            if not os.path.exists(feature_path):
                with open(feature_path, mode="w", newline="") as f:
                    for val in ["class"] + landmarks_right_hand[1:]:
                        print(f"Writing: {val}")
                        f.write(val + "\n")
                  
            if not os.path.exists(self.get_csv_path_right_hand()):
                with open(self.get_csv_path_right_hand(), mode="w", newline="") as f:
                    csv_writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    csv_writer.writerow(landmarks_right_hand)

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

    def train_data(self):
        category = ""
        if self.exportingLeftHand:
            category = "left_hand"
        if self.exportingRightHand:
            category = "right_hand"
        if self.exportingBodyPose:
            category = "body_pose"
        if category == "":
            return
        pattern = f"{category}_{{class}}.txt"
        all_files = os.listdir(CSV_DIR)
        relevant_files = [f for f in all_files if f.startswith(category)]
        combined_data = pd.concat(
            [pd.read_csv(os.path.join(CSV_DIR, f)) for f in relevant_files],
            ignore_index=True,
        )
        combined_data.to_csv("combined_data.csv", index=False)
        # Combine data first

        # If no data was returned (because no files were found for the category), exit the function
        if combined_data is None:
            print(f"Skipping training for category {category} due to lack of data.")
            return

        X = combined_data.drop("class", axis=1)
        y = combined_data["class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1234
        )
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        pipelines = {
            "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
        }

        fit_models = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X_train, y_train)
            fit_models[algo] = model

        for algo_name, trained_model in fit_models.items():
            model_path = os.path.join(MODEL_DIR, f"{category}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(trained_model, f)

    def predict_model(self, image, results):
        # pdb.set_trace()
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
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                class_index = list(model.classes_).index(body_language_class)
                prob_of_predicted_class = body_language_prob[class_index]
                if model_name == "left_hand" and prob_of_predicted_class * 100 > 70:
                    print(
                        f"Predicted Class: {body_language_class}, Probability: {prob_of_predicted_class*100}%"
                    )
                    coords = tuple(
                        np.multiply(
                            np.array(
                                (
                                    results.left_hand_landmarks.landmark[0].x,  # Index 0 corresponds to the left wrist in the hand landmarks
                                    results.left_hand_landmarks.landmark[0].y,
                                )
                            ),
                            [640, 480],
                        ).astype(int)
)

                    # Calculate text width and height with the new font size
                    (text_width, text_height), _ = cv2.getTextSize(body_language_class + str(round(body_language_prob[np.argmax(body_language_prob)], 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)

                    # Adjust padding if needed. Here, I'm reducing the padding since the font is smaller.
                    padding = 5

                    # Draw the rectangle based on the new text size
                    cv2.rectangle(
                        image,
                        (coords[0], coords[1] + padding),
                        (coords[0] + text_width + padding, coords[1] - text_height - padding),
                        (245, 117, 16),
                        -1,
                    )

                    # Draw the text with adjusted font size
                    cv2.putText(
                        image,
                        body_language_class + str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                        coords,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,  # Adjusted font size to 0.7
                        (255, 255, 255),
                        1,  # Adjusted line thickness to 1 for smaller text
                        cv2.LINE_AA,
                    )                     
                   

                if model_name == "right_hand" and prob_of_predicted_class * 100 > 70:
                        print(
                            f"Predicted Class: {body_language_class}, Probability: {prob_of_predicted_class*100}%"
                        )
                        coords = tuple(
                            np.multiply(
                                np.array(
                                    (
                                        results.right_hand_landmarks.landmark[0].x,  # Index 0 corresponds to the left wrist in the hand landmarks
                                        results.right_hand_landmarks.landmark[0].y,
                                    )
                                ),
                                [640, 480],
                            ).astype(int)
    )

                        # Calculate text width and height with the new font size
                        (text_width, text_height), _ = cv2.getTextSize(body_language_class + str(round(body_language_prob[np.argmax(body_language_prob)], 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)

                        # Adjust padding if needed. Here, I'm reducing the padding since the font is smaller.
                        padding = 5

                        # Draw the rectangle based on the new text size
                        cv2.rectangle(
                            image,
                            (coords[0], coords[1] + padding),
                            (coords[0] + text_width + padding, coords[1] - text_height - padding),
                            (245, 117, 16),
                            -1,
                        )

                        # Draw the text with adjusted font size
                        cv2.putText(
                            image,
                            body_language_class + str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                            coords,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,  # Adjusted font size to 0.7
                            (255, 255, 255),
                            1,  # Adjusted line thickness to 1 for smaller text
                            cv2.LINE_AA,
                        )

                if model_name == "right_hand" and prob_of_predicted_class * 100 > 70:
                        print(
                            f"Predicted Class: {body_language_class}, Probability: {prob_of_predicted_class*100}%"
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


# saddasd
