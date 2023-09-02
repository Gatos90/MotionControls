from tkinter import *
import cv2
from PIL import Image, ImageTk
from motion_tracking import Track_Object
import time
import threading
from queue import Queue  # Import the Queue class


MAX_QUEUE_SIZE = 10


class VideoCaptureThread(threading.Thread):
    def __init__(self, queue, callback_fps):  # Add queue as a parameter
        super().__init__()
        self.queue = queue  # Assign the queue to an instance variable
        self.callback_fps = callback_fps
        self.tracker = Track_Object()

    def run(self):
        video_capture = cv2.VideoCapture(1)
        prev_frame_time = 0
        new_frame_time = 0
        while True:
            _, frame = video_capture.read()
            tracked_frame = self.tracker.process_image(frame)

            # Check if tracked_frame is None before attempting to continue
            if tracked_frame is None:
                continue

            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            self.callback_fps(int(fps))

            if self.queue.qsize() < MAX_QUEUE_SIZE:
                self.queue.put(
                    ("frame", tracked_frame)
                )  # Put the tracked frame into the queue


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.X_train = None

        self.event_log = Listbox(self, height=8, width=50)
        self.scrollbar = Scrollbar(self, orient="vertical")
        self.event_log.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.event_log.yview)

        self.scrollbar.pack(side="right", fill="y")
        self.event_log.pack(side="left")

        self.image_label = Label(self)
        self.image_label.pack()

        self.queue = Queue()  # Create a queue instance
        self.thread = VideoCaptureThread(
            self.queue, self.update_fps
        )  # Pass the queue to the thread
        self.thread.start()

        self.show_video = True

        self.class_name_label = Label(self, text="Class Name:")
        self.class_name_label.pack(pady=5)  # add some padding for aesthetics

        self.class_name_entry = Entry(self)
        self.class_name_entry.pack(pady=5)

        self.start_export_button = Button(
            self, text="Start Export", command=self.set_class_and_start_export
        )
        self.start_export_button.pack()

        self.toggle_button = Button(
            self, text="Toggle Video", command=self.thread.tracker.should_show_video
        )
        self.toggle_button.pack()

        self.stop_export_button = Button(
            self, text="Stop Export", command=self.thread.tracker.stop_export
        )
        self.stop_export_button.pack()

        self.train_model_button = Button(
            self,
            text="Train and Save Model",
            command=self.thread.tracker.start_training,
        )
        self.train_model_button.pack()

        self.after(100, self.check_queue)  # Start checking the queue

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

    def set_class_and_start_export(self):
        class_name = self.class_name_entry.get()  # Get the entered class name
        self.thread.tracker.set_class_name(class_name)
        self.thread.tracker.start_export()


if __name__ == "__main__":
    root = Tk()
    app = Window(root)
    root.wm_title("Tkinter window")
    root.geometry("600x450")
    root.mainloop()
