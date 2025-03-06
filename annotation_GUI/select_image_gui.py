import os
import requests
import json
import sys

import tkinter as tk
from tkinter import messagebox, Listbox, Scrollbar

import pandas as pd
import numpy as np

# Set up file processing
patients = []
files = []
parts = []

folder_ct = sys.argv[1]

with os.scandir(folder_ct) as entries:
    for entry in entries:
        if ".nii.gz" in entry.name:
            files.append(entry.name)
            patients.append(entry.name.split('_')[0])

            if "H-N" in entry.name:
                parts.append('H')

            if "TORSO" in entry.name:
                parts.append('T')

data = {'patients': patients, 'files': files, 'parts': parts}
print(data)
df = pd.DataFrame(data)
file_lists = np.concatenate(df.groupby('patients').agg({'parts': np.sort, 'files': list})['files'].values)

# Function to execute command
def execute_command():
    user_id = userid_entry.get().strip()
    if not user_id:
        messagebox.showwarning("Empty Entry", "Please enter a valid user ID before executing.")
        return

    # Clear Listbox
    listbox.delete(0, tk.END)

    for i, file_name in enumerate(file_lists):
        os.system(f"python ct_viewer_gui.py {os.path.join(folder_ct, file_name)} {user_id}")

    messagebox.showinfo("Completed", "Processing completed!")

# GUI Setup
root = tk.Tk()
root.title("CT File Processor")

# Set window size
window_width = 600
window_height = 200

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate position for center alignment
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Set geometry (width x height + x_offset + y_offset)
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

root.configure(bg="#f4f4f4")  # Light gray background

# Title Label
title_label = tk.Label(root, text="CT File Processor", font=("Helvetica", 16, "bold"), bg="#f4f4f4", fg="#333")
title_label.pack(pady=10)

# Frame for Input
frame = tk.Frame(root, bg="#f4f4f4")
frame.pack(pady=10)

# Label and Entry for User ID
userid_label = tk.Label(frame, text="User ID:", font=("Helvetica", 12), bg="#f4f4f4")
userid_label.grid(row=0, column=0, padx=5, pady=5)
userid_entry = tk.Entry(frame, width=40, font=("Arial", 12))
userid_entry.grid(row=0, column=1, padx=5, pady=5)

# Button to Execute
execute_btn = tk.Button(root, text="Start Processing", font=("Helvetica", 12, "bold"), bg="#007BFF", fg="white",
                        width=20, height=2, command=execute_command)
execute_btn.pack(pady=10)

# Listbox to Show Processed Files
listbox_frame = tk.Frame(root, bg="#f4f4f4")
listbox_frame.pack(pady=10, fill="both", expand=True)

listbox = Listbox(listbox_frame, width=80, height=10, font=("Helvetica", 10))
listbox.pack(side="left", fill="both", expand=True)

# Scrollbar for Listbox
scrollbar = Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
scrollbar.pack(side="right", fill="y")
listbox.config(yscrollcommand=scrollbar.set)

# Run Tkinter
root.mainloop()
