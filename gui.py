# gui.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from motion_tracking import MotionTracker

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the main window to the size of the display
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.setGeometry(0, 0, screen_size.width(), screen_size.height())

        # Create a central widget and layout for the main window
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Create a QLabel to simulate a video display
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(300, 300)
        self.video_label.setAlignment(Qt.AlignCenter)
        # For demonstration purposes, set a red background to the QLabel
        self.video_label.setStyleSheet("background-color: red;")

        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.motion_tracker = MotionTracker()

        # Create a QTimer to update the video frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)  # Update every 30 ms

    def update_frame(self):
        frame = self.motion_tracker.get_frame()
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.motion_tracker.release()
        event.accept()

    def resizeEvent(self, event):
        # Resize the video_label proportionally with the main window
        width_ratio = self.width() / self.screen().size().width()
        height_ratio = self.height() / self.screen().size().height()

        new_width = int(300 * width_ratio)  # Convert to integer
        new_height = int(300 * height_ratio)   # Convert to integer

        self.video_label.setFixedSize(new_width, new_height)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
