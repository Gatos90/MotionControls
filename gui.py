from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import cv2
import numpy as np
from motion_tracking import Track_Object
import time  # Added for FPS calculation


class ThreadClass(QThread):
    send_image = pyqtSignal(np.ndarray)
    FPS = pyqtSignal(int)  # Signal for FPS

    def run(self):
        video_capture = cv2.VideoCapture(0)
        tracker = Track_Object()
        prev_frame_time = 0
        new_frame_time = 0
        while True:
            _, frame = video_capture.read()
            tracked_frame = tracker.process_image(frame)
            
            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            self.FPS.emit(int(fps))
            
            if tracked_frame is not None:
                self.send_image.emit(tracked_frame)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        self.thread = ThreadClass()
        self.thread.send_image.connect(self.update_image)
        self.thread.FPS.connect(self.update_fps)  # Connect the FPS signal
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        qt_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(int)
    def update_fps(self, fps):
        # You can update a label or any other widget with the FPS value here.
        pass


if __name__ == "__main__":
    app = QApplication([])
    win = Window()
    win.show()
    app.exec_()
