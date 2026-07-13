# Digit Reader

## About

Digit Reader is a GUI application that allows users to draw handwritten digits on a canvas and receive a prediction from a machine-learning model. The application can detect and predict multiple non-touching digits, then combine the individual predictions into a single number.

## Libraries Used

- **CustomTkinter** — renders the application's graphical interface.
- **NumPy** — provides numerical operations and array manipulation.
- **Matplotlib** — visualises model-training and evaluation results.
- **OpenCV** and **Pillow** — process and transform images.
- **Joblib** — saves and loads the trained model.
- **Scikit-learn** — preprocesses the training data and trains and evaluates the model.

## About the Model

The application uses a K-nearest neighbours (KNN) classifier. KNN predicts a digit by finding the most similar labelled examples in its training data and using their classes to determine the result.

### Dataset

The model was trained on the [MNIST dataset](https://www.openml.org/d/554), which contains 28 × 28 grayscale images of handwritten digits from 0 to 9.

### Training Preprocessing

Before training, the image data passes through a Scikit-learn pipeline containing two components:

1. `MinMaxScaler` converts pixel values from the range 0–255 to 0–1.
2. `ShadeRemover`, a custom transformer, converts the grayscale images into binary images. Pixels at or above its threshold become `1`, while pixels below the threshold become `0`.

The training and evaluation process is available in [`externals/model.ipynb`](externals/model.ipynb).

## Model Evaluation

Several metrics were used to evaluate the classifier. The results below are rounded to three significant figures:

| Metric | Result | Description |
| --- | ---: | --- |
| Precision | 96.6% | The proportion of predicted instances that were classified correctly. |
| Recall | 96.6% | The proportion of actual instances that were identified correctly. |
| Weighted F1 score | 96.6% | A balance of precision and recall, weighted by the number of examples in each class. |
| ROC AUC score | 99.3% | The model's ability to distinguish between classes across different classification thresholds. |

## Application Preprocessing

When the user selects **Get Number**, the drawing passes through the following pipeline before prediction:

1. The Pillow canvas image is converted into an OpenCV grayscale image.
2. The pixel values are normalised and inverted so that the background is `0` and the digit strokes are `1`, matching the model's training data.
3. OpenCV identifies each separate digit as a connected component:

   ```python
   cv2.connectedComponentsWithStats(image)
   ```

4. Each detected digit is cropped, resized while preserving its aspect ratio, and centred in a 28 × 28 image.
5. Each image is flattened from a 28 × 28 array into 784 features.
6. The model predicts every digit, and the application combines the results into a single number.

## Project Structure

```text
digit_reader/
├── main.py                   # Application entry point
├── src/
│   ├── gui.py                # Drawing interface and user interaction
│   └── model.py              # Image preprocessing and classification
├── models/
│   └── knn_classifier.joblib # Trained KNN classifier
├── externals/
│   └── model.ipynb           # Model training and evaluation notebook
└── README.md
```

## Installation

Clone the repository and navigate to its directory:

```bash
git clone <repository-url>
cd digit_reader
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, activate the environment with:

```powershell
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install customtkinter pillow numpy opencv-python scikit-learn joblib matplotlib
```

Make sure the trained model is located at:

```text
models/knn_classifier.joblib
```

## Running the Application

Start the application from the project directory:

```bash
python3 main.py
```

A window containing the drawing canvas will open. Draw one or more separated digits, select **Get Number** to classify them, or select **Clear** to reset the canvas.

## Current Limitations

- Digits must not touch because each connected region is treated as a separate digit.
- Small strokes may be discarded as image noise.
- Digits are ordered from left to right; multi-row input is not currently supported.
- Recognition accuracy depends on how closely the processed drawings resemble the MNIST training images.
