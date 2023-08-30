from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
from motion_tracking import Track_Object
import time
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

        self.event_log = Listbox(self, height=8, width=50)
        self.scrollbar = Scrollbar(self, orient="vertical")
        self.event_log.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.event_log.yview)

        self.scrollbar.pack(side="right", fill="y")
        self.event_log.pack(side="left")

        self.toggle_button = Button(
            self, text="Toggle Video", command=self.toggle_video
        )
        self.toggle_button.pack()

        self.image_label = Label(self)
        self.image_label.pack()

        self.thread = None

        self.show_video = True

    def start_thread(self):
        self.thread = VideoCaptureThread(self.update_image, self.update_fps)
        self.thread.start()

    def update_image(self, cv_img):
        if self.show_video:
            img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            self.image_label.config(image=imgtk)
            self.image_label.image = imgtk

    def toggle_video(self):
        self.show_video = not self.show_video
        if self.show_video:
            self.image_label.pack()
        else:
            self.image_label.pack_forget()
            self.log_event(f"Video toggled {'on' if self.show_video else 'off'}")

    def update_fps(self, fps):
        self.log_event(f"FPS updated to {fps}")

    def log_event(self, event):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.event_log.insert(0, f"{timestamp}: {event}")


if __name__ == "__main__":
    root = Tk()
    app = Window(root)
    app.start_thread()
    root.wm_title("Tkinter window")
    root.geometry("600x450")
    root.mainloop()
