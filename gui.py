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
frame_label.pack()

thread = threading.Thread(target=stream, args=(frame_label,))
thread.daemon = True
thread.start()

root.mainloop()
