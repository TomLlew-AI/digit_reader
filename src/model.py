from sklearn.datasets import fetch_openml
from PIL import Image, ImageDraw
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

class Preprocessor():
    def __init__(self, image: Image.Image):
        # OpenCV represents grayscale images as 2D uint8 NumPy arrays.
        rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        self.image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        self.digit_groups = None

    def standardize_image(self, image: np.ndarray):
        new_image = 1.0 - image.astype(np.float32) / 255
        return new_image.astype(np.int8)

    def separate_digits(self, image):
        # Make digits white and background black

        count, labels, stats, _ = cv2.connectedComponentsWithStats(image)

        digit_groups = []

        # Component 0 is the background
        for component in range(1, count):
            x, y, width, height, area = stats[component]

            # Ignore small accidental marks
            if area < 300:
                continue

            digit = image[y:y + height, x:x + width]
            digit_groups.append((x, digit))

        # Arrange digits from left to right
        digit_groups.sort(key=lambda item: item[0])

        return [digit for _, digit in digit_groups]

    def reshape_digit_groups(self, digit_groups):

        target_size = 28
        inner_size = 20

        results = np.zeros(
            (len(digit_groups), target_size, target_size),
            dtype=np.uint8
        )

        for i, group in enumerate(digit_groups):
            group = group.astype(np.uint8)

            height, width = group.shape

            # Preserve the original aspect ratio
            scale = min(
                inner_size / width,
                inner_size / height
            )

            resized_width = max(1, round(width * scale))
            resized_height = max(1, round(height * scale))

            resized = cv2.resize(
                group,
                (resized_width, resized_height),
                interpolation=cv2.INTER_NEAREST
            )

            # Center the digit inside the 28×28 image
            x_offset = (target_size - resized_width) // 2
            y_offset = (target_size - resized_height) // 2

            results[
                i,
                y_offset:y_offset + resized_height,
                x_offset:x_offset + resized_width
            ] = resized

        return results


    def run_pipeline(self):
        standardized_image = self.standardize_image(self.image)
        digit_groups = self.separate_digits(standardized_image)
        reshaped_digit_groups = self.reshape_digit_groups(digit_groups)

        self.digit_groups = reshaped_digit_groups
        return self.digit_groups


class Classifier():
    def __init__(self, classifier_path=None):
        self.classifier_path = classifier_path
        if classifier_path != None:
            self.load_classifier(classifier_path)

    def load_classifier(self, classifier_path):
        self.classifier_path = classifier_path
        self.classifier = joblib.load(classifier_path)

    def predict_digits(self, images: np.ndarray):
        new_images = images.reshape((len(images), len(images[0].flatten())))
        predicted = self.classifier.predict(new_images)
        return predicted.astype(np.int32)

mnist = fetch_openml('mnist_784', version=1)
