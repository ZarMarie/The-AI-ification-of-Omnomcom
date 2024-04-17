import tkinter as tk
from tkinter import messagebox
import threading

image_path = 'data/AppleJuice/1.png'
price = int


def show_custom_dialog(image_path, price):
    # Create a new window for the dialog
    top = tk.Toplevel()
    top.title("Is this your item?")

    # Load the image
    image = tk.PhotoImage(file=image_path)  # Replace 'image_path' with your actual image path

    # Label to display the image
    image_label = tk.Label(top, image=image)
    image_label.pack()

    # Text label for the question
    text_label = tk.Label(top, text=f"Is this your item? Selling for {price} Euros (€)")
    text_label.pack()

    # Function to handle button clicks
    def button_click(result):
        top.destroy()  # Close the dialog window
        if result:
          print("User clicked Yes.")
        else:
          print("User clicked No.")

    # Buttons for Yes and No
    yes_button = tk.Button(top, text="Yes", command=lambda: button_click(True))
    yes_button.pack(side=tk.LEFT, padx=150, pady=25)
    no_button = tk.Button(top, text="No", command=lambda: button_click(False))
    no_button.pack(side=tk.RIGHT, padx=150, pady=25)

    # Run the main loop for the dialog window
    top.mainloop()
