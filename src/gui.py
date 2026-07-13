import customtkinter as ctk
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from src.model import Classifier, Preprocessor
import matplotlib.pyplot as plt


# Constants
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 1000
WINDOW_BG_COLOR = "#CCCCCC"

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=WINDOW_BG_COLOR)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.title("Digits Drawer")

        self.drawing_frame_width = 1200
        self.drawing_frame_height = 800
        self.pen_width = 30

        model_path = "models/knn_classifier.joblib"
        self.digitModel = Classifier(model_path)

        self.render()

    def render(self):
        self.drawing_frame = ctk.CTkFrame(
            self,
            width=self.drawing_frame_width,
            height=self.drawing_frame_height,
            fg_color="#F5F5F5",
            corner_radius=0,
            border_color="#000000",
            border_width=2
        )

        self.drawing_frame.place(anchor='center', x=20 + self.drawing_frame_width // 2, y=WINDOW_HEIGHT // 2)

        self.canvas = tk.Canvas(self.drawing_frame,
                                width=self.drawing_frame_width,
                                height=self.drawing_frame_height,
                                bg='white',
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Pillow image to mirror drawings (grayscale)
        self.image = Image.new('L', (self.drawing_frame_width, self.drawing_frame_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Mouse state
        self.last_x = None
        self.last_y = None

        # Bind mouse events to canvas
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Controls
        controls = ctk.CTkFrame(self, fg_color=WINDOW_BG_COLOR, corner_radius=0)
        controls.place(x=20, y=20)

        clear_btn = ctk.CTkButton(
            controls,
            text="Clear",
            font=("Arial", 20),
            fg_color="#666666",
            hover_color="#444444",
            command=self.clear_canvas)
        clear_btn.grid(row=0, column=0, padx=6, pady=6, ipady=5)

        get_pred_btn = ctk.CTkButton(
            controls,
            text="Get Number",
            font=("Arial", 20),
            fg_color="#666666",
            hover_color="#444444",
            command=self.get_prediction
        )
        get_pred_btn.grid(row=0, column=1, padx=6, pady=6, ipady=5)

        self.prediction_text = ctk.CTkLabel(
            controls,
            text="",
            font=("Arial", 20),
            text_color="#444444"
        )
        self.prediction_text.grid(row=0, column=2, padx=6, pady=6, ipady=5)

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_move(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # draw on canvas (visual)
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.pen_width, fill='black',
                                    capstyle=ctk.ROUND, smooth=True)
            # draw on PIL image (for processing / saving)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=self.pen_width)

        self.last_x, self.last_y = x, y

    def on_button_release(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.drawing_frame_width, self.drawing_frame_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_text.configure(text="")

    def int_index_to_arr(self, index, width):
        return [index // width, index % width]

    def arr_index_to_int(self, arr, width):
        return arr[0] * width + arr[1]

    def get_prediction(self):

        # Preprocess the image by resizing, and mapping values
        preprocessor = Preprocessor(self.image)
        digit_groups = preprocessor.run_pipeline()

        # Predict the digits using the trained model
        predictions = self.digitModel.predict_digits(digit_groups)

        str_predictions = ''.join([str(s) for s in predictions])
        self.prediction_text.configure(text=str_predictions)
