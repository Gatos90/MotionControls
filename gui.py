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
    def __init__(self, queue, callback_fps):  # Add queue as a parameter
        super().__init__()
        self.queue = queue  # Assign the queue to an instance variable
        self.callback_fps = callback_fps
        self.tracker = Track_Object()

    def run(self):
        video_capture = cv2.VideoCapture(1)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
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
        
        notebook = ttk.Notebook(self)  # Create a notebook object

        # Create frames to act as the contents of your tabs
        tab1 = Frame(notebook)  
        tab2 = Frame(notebook)

        # Add these frames to your notebook
        notebook.add(tab1, text="Train a Model")
        notebook.add(tab2, text="Your Models")

        # Place the notebook itself
        notebook.pack(expand=1, fill="both")

        self.left_hand_var = BooleanVar()
        self.right_hand_var = BooleanVar()
        self.body_var = BooleanVar()

        self.event_log = Listbox( tab1, height=8, width=50)
        self.scrollbar = Scrollbar( tab1, orient="vertical")
        self.event_log.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.event_log.yview)

        self.scrollbar.pack(side="right", fill="y")
        self.event_log.pack(side="left")

        self.image_label = Label( tab1,)
        self.image_label.pack()

        self.class_name_var = StringVar()
        self.class_name_var.trace("w", self.validate_class_name)

        self.class_name_entry = Entry( tab1, textvariable=self.class_name_var)
        self.class_name_entry.pack(pady=5)

        self.start_export_button = Button(
             tab1,
            text="Start Export",
            command=self.set_class_and_start_export,
            state=DISABLED,
        )
        self.start_export_button.pack()

        self.queue = Queue() 
        self.thread = VideoCaptureThread(
            self.queue, self.update_fps
        )  
        self.thread.start()

        self.show_video = True

        self.class_name_label = Label( tab1, text="Class Name:")
        self.class_name_label.pack(pady=5)  

        self.left_hand_cb = Checkbutton(
            tab1,
            text="Left Hand",
            variable=self.left_hand_var,
            command=self.update_checkbox,
        )
        self.left_hand_cb.pack()

        self.right_hand_cb = Checkbutton(
            tab1,
            text="Right Hand",
            variable=self.right_hand_var,
            command=self.update_checkbox,
        )
        self.right_hand_cb.pack()

        self.body_cb = Checkbutton(
             tab1, text="Body", variable=self.body_var, command=self.update_checkbox
        )
        self.body_cb.pack()

        self.start_export_button.pack()

        self.toggle_button = Button(
             tab1, text="Toggle Video", command=self.thread.tracker.should_show_video
        )
        self.toggle_button.pack()

        self.stop_export_button = Button(
             tab1, text="Stop Export", command=self.thread.tracker.stop_export
        )
        self.stop_export_button.pack()

        self.train_model_button = Button(
            tab1,
            text="Train and Save Model",
            command=self.thread.tracker.start_training,
        )
        self.train_model_button.pack()

        self.after(100, self.check_queue)   
      
        self.update_file_lists_button = Button(tab2, text="Update File Lists", command=self.update_file_lists)
        self.update_file_lists_button.pack()

        self.body_pose_listbox = Listbox(tab2)
        self.left_hand_listbox = Listbox(tab2)
        self.right_hand_listbox = Listbox(tab2)
                
        self.body_pose_listbox.pack()
        self.left_hand_listbox.pack()
        self.right_hand_listbox.pack()

        self.update_file_lists()


    def display_gif(self, gif_path, selected_file):
        try:
            im = Image.open(gif_path)
            self.frames = []
            self.gif_label = Label(self.master)
            self.gif_label.pack(side="right", padx=10)
            
            im = Image.open(gif_path)
            sequence = [ImageTk.PhotoImage(im_frame) for im_frame in ImageSequence.Iterator(im)]
            self.frames = sequence
            self.current_frame = 0
            self.gif_label.config(image=self.frames[self.current_frame])
            self.master.after(100, self.update_gif)
        except FileNotFoundError:
            print(f"{gif_path} not found. Generating...")
            read_csv_and_draw(selected_file)  # Assuming this is the method that generates the gif
            try:
                im = Image.open(gif_path)
                # rest of your code to display gif
            except Exception as e:
                print(f"An error occurred: {e}")    

    def update_gif(self):
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.gif_label.config(image=self.frames[self.current_frame])
        self.master.after(100, self.update_gif)

    
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

    def validate_class_name(self, *args):
        class_name = self.class_name_entry.get()
        if class_name:
            self.start_export_button.config(state=NORMAL)  # Enable the button
        else:
            self.start_export_button.config(state=DISABLED)


if __name__ == "__main__":
    root = Tk()
    app = Window(root)
    root.wm_title("Tkinter window")
    root.geometry("600x450")
    root.mainloop()
