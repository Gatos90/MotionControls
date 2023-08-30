from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
from motion_tracking import Track_Object
import time  # Added for FPS calculation
import threading


class VideoCaptureThread(threading.Thread):
    def __init__(self, callback_image, callback_fps):
        super().__init__()
        self.callback_image = callback_image
        self.callback_fps = callback_fps

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
            self.callback_fps(int(fps))

            if tracked_frame is not None:
                self.callback_image(tracked_frame)


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.image_label = Label(self)
        self.image_label.pack()

        self.thread = VideoCaptureThread(self.update_image, self.update_fps)
        self.thread.start()

    def update_image(self, cv_img):
        cv2image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk

    def update_fps(self, fps):
        # You can update a label or any other widget with the FPS value here.
        pass


if __name__ == "__main__":
    root = Tk()
    app = Window(root)
    root.wm_title("Tkinter window")
    root.geometry("600x450")
    root.mainloop()
