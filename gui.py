# gui.py

import tkinter as tk
from PIL import Image, ImageTk
from motion_tracking import videofeed
import threading

def stream(label):
    frame = videofeed()
    if frame is not None:
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, lambda: stream(label))


root = tk.Tk()
frame_label = tk.Label(root)
frame_label.grid(row=0, column=0)  # Position at row 0, column 0

text_label = tk.Label(root, text="Hello")
text_label.grid(row=1, column=1)  # Position at row 1, column 1

thread = threading.Thread(target=stream, args=(frame_label,))
thread.daemon = 1
thread.start()

root.mainloop()

