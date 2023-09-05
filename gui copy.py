import cv2
import os
import threading
import time
from queue import Queue
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tkinter import *
import cv2
from PIL import Image, ImageTk
from motion_tracking import Track_Object
import time
import threading
from queue import Queue  # Import the Queue class
import csv
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import imageio.v2 as imageio
import os
from tkinter import ttk
from PIL import ImageSequence



MAX_QUEUE_SIZE = 10


def list_csv_files_by_sector(directory="models/cords/"):
    body_pose_files = []
    left_hand_files = []
    right_hand_files = []
    
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            if filename.startswith("body_pose"):
                body_pose_files.append(os.path.join(directory, filename))
            elif filename.startswith("left_hand"):
                left_hand_files.append(os.path.join(directory, filename))
            elif filename.startswith("right_hand"):
                right_hand_files.append(os.path.join(directory, filename))
                
    return body_pose_files, left_hand_files, right_hand_files


def resize_and_center(image, target_size=(300, 300)):
    height, width = image.shape[:2]
    scale_factor = min(target_size[0] / width, target_size[1] / height)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height))

    top = (target_size[1] - new_height) // 2
    bottom = target_size[1] - top - new_height
    left = (target_size[0] - new_width) // 2
    right = target_size[0] - left - new_width

    centered_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    return centered_image


def read_csv_and_draw(selected_file):
    csv_file = selected_file
    gif_file = os.path.splitext(os.path.basename(csv_file))[0] + ".gif"
    image_list = []

    min_x, max_x, min_y, max_y = (
        float("inf"),
        float("-inf"),
        float("inf"),
        float("-inf"),
    )

    # First pass to find global bounding box
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords = [
                (float(row.get(f"x{i+1}", 0.0)), float(row.get(f"y{i+1}", 0.0)))
                for i in range(33)
            ]
            min_x = min(min_x, min(x for x, _ in coords) * 250)
            max_x = max(max_x, max(x for x, _ in coords) * 250)
            min_y = min(min_y, min(y for _, y in coords) * 250)
            max_y = max(max_y, max(y for _, y in coords) * 250)

    global_width = int(max_x - min_x)
    global_height = int(max_y - min_y)

    # Second pass to actually create images
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            coords = [
                (float(row.get(f"x{i+1}", 0.0)), float(row.get(f"y{i+1}", 0.0)))
                for i in range(33)
            ]

            img = np.zeros((global_height, global_width, 3), dtype=np.uint8)
            coords_pixel = [
                (int((x - min_x / 250) * 250), int((y - min_y / 250) * 250))
                for x, y in coords
            ]

            for x, y in coords_pixel:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            # Add line counter text
            counter_text = f"Line: {index + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (10, 30)
            font_scale = 0.8
            font_color = (255, 255, 255)
            line_type = 2

            cv2.putText(
                img, counter_text, position, font, font_scale, font_color, line_type
            )

            img = resize_and_center(img)

            filename = f"models/images/cache/image_{index}.png"
            cv2.imwrite(filename, img)
            image_list.append(filename)

        images = [imageio.imread(image) for image in image_list]
        imageio.mimsave(f"models/images/{gif_file}", images, duration=0.5)

        # Delete the PNG files
        for image in image_list:
            os.remove(image)


class VideoCaptureThread(threading.Thread):
    def __init__(self, queue, callback_fps):
        super().__init__()
        self.queue = queue
        self.callback_fps = callback_fps

    def run(self):
        video_capture = cv2.VideoCapture(1)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        while True:
            _, frame = video_capture.read()

            if self.queue.qsize() < MAX_QUEUE_SIZE:
                self.queue.put(("frame", frame))


class App(QMainWindow):
    update_image_signal = pyqtSignal(QImage)

    def __init__(self):
        super(App, self).__init__()
        self.queue = Queue()
        self.thread = VideoCaptureThread(self.queue, self.update_fps)
        self.thread.start()
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout() # Initializing `self.layout`

        self.tab_train_layout = QVBoxLayout()
        self.tab_models_layout = QVBoxLayout()

        self.body_pose_listbox = QListWidget(self)
        self.left_hand_listbox = QListWidget(self)
        self.right_hand_listbox = QListWidget(self)

        self.update_file_lists_button = QPushButton("Update File Lists")
        self.update_file_lists_button.clicked.connect(self.update_file_lists)

        self.tab_models_layout.addWidget(self.update_file_lists_button)
        self.tab_models_layout.addWidget(self.body_pose_listbox)
        self.tab_models_layout.addWidget(self.left_hand_listbox)
        self.tab_models_layout.addWidget(self.right_hand_listbox)


        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.tab_train = QWidget()
        self.tab_models = QWidget()

        self.tabs.addTab(self.tab_train, "Train a Model")
        self.tabs.addTab(self.tab_models, "Your Models")

        self.tab_train.setLayout(self.tab_train_layout)
        self.tab_models.setLayout(self.tab_models_layout)

        self.label = QLabel()
        self.layout.addWidget(self.label)

        self.textbox = QLineEdit(self)
        self.layout.addWidget(self.textbox)

        self.train_button = QPushButton("Toggle Video")
        self.layout.addWidget(self.train_button)

        self.train_button = QPushButton("Start Export")
        self.layout.addWidget(self.train_button)

        self.train_button = QPushButton("Stop Export")
        self.layout.addWidget(self.train_button)

        self.checkbox = QCheckBox("Left Hand")
        self.layout.addWidget(self.checkbox)

        self.checkbox = QCheckBox("Right Hand")
        self.layout.addWidget(self.checkbox)

        self.checkbox = QCheckBox("Body")
        self.layout.addWidget(self.checkbox)

        self.list_widget = QListWidget(self)
        self.layout.addWidget(self.list_widget)

        self.train_button = QPushButton("Train and Save Model")
        self.layout.addWidget(self.train_button)

        self.central_widget.setLayout(self.layout)

        self.setWindowTitle("Your App")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)

        self.show()


    def update_image(self):
        while not self.queue.empty():
            item_type, item_data = self.queue.get()
            if item_type == "frame":
                self.display_frame(item_data)

    def display_frame(self, frame):
        qt_image = self.convert_to_qt_image(frame)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def convert_to_qt_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        return qt_image

    def update_fps(self, fps):
        pass  # Implement your function to update fps here

    def update_checkbox(self):
        is_left_hand_checked = self.left_hand_var.get()
        is_right_hand_checked = self.right_hand_var.get()
        is_body_checked = self.body_var.get()

        self.thread.tracker.set_export_left_hand(
            True if is_left_hand_checked else False
        )
        self.thread.tracker.set_export_right_hand(
            True if is_right_hand_checked else False
        )
        self.thread.tracker.set_export_body_pose(True if is_body_checked else False)
        
    def update_file_lists(self):
        body_pose_files, left_hand_files, right_hand_files = list_csv_files_by_sector()

        self.body_pose_listbox.delete(0, END)
        self.left_hand_listbox.delete(0, END)
        self.right_hand_listbox.delete(0, END)

        for file in body_pose_files:
            self.body_pose_listbox.insert(END, file)
        
        for file in left_hand_files:
            self.left_hand_listbox.insert(END, file)

        for file in right_hand_files:
            self.right_hand_listbox.insert(END, file)    
            
        self.body_pose_listbox.bind('<<ListboxSelect>>', self.on_body_pose_selected)
        
        
    def on_body_pose_selected(self, event):
        selected_index = self.body_pose_listbox.curselection()[0]
        selected_file = self.body_pose_listbox.get(selected_index)
        gif_file = os.path.splitext(os.path.basename(selected_file))[0] + ".gif"
        print(f"models/images/{gif_file}")
        self.display_gif(f"models/images/{gif_file}", selected_file)  
          
    

    def check_queue(self):
        while not self.queue.empty():
            item_type, item_data = self.queue.get()
            if item_type == "frame":
                self.update_image(item_data)
            elif item_type == "fps":
                self.log_event(
                    f"FPS updated to {item_data}"
                )  # Update FPS in the main thread
        self.after(100, self.check_queue)  # Keep checking the queue    
        
    def set_class_and_start_export(self):
        class_name = self.class_name_entry.get()  # Get the entered class name
        self.thread.tracker.set_class_name(class_name)
        self.thread.tracker.start_export()

    def validate_class_name(self, *args):
        class_name = self.class_name_entry.get()
        if class_name:
            self.start_export_button.config(state=NORMAL)  # Enable the button
        else:
            self.start_export_button.config(state=DISABLED)        
        
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())