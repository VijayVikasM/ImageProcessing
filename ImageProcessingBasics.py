# image_processing_functions.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(image_path, output_path="gray_image.png"):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray_image)

def resize_image(image_path, width, height, output_path="resized_image.png"):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(output_path, resized_image)

def rotate_90(image_path, output_path="rotated_90_image.png"):
    image = cv2.imread(image_path)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(output_path, rotated_image)

def flip_horizontal(image_path, output_path="flipped_image.png"):
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, 1)
    cv2.imwrite(output_path, flipped_image)

def increase_brightness(image_path, value=50, output_path="bright_image.png"):
    image = cv2.imread(image_path)
    bright_image = cv2.convertScaleAbs(image, alpha=1, beta=value)
    cv2.imwrite(output_path, bright_image)

def gaussian_blur(image_path, output_path="blurred_image.png"):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite(output_path, blurred_image)

def binary_threshold(image_path, output_path="binary_image.png"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, binary_image)

def histogram(image_path, output_path="histogram.png"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(output_path)

def crop_image(image_path, x, y, w, h, output_path="cropped_image.png"):
    image = cv2.imread(image_path)
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_image)

def edge_detection(image_path, output_path="edges.png"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    cv2.imwrite(output_path, edges)

def draw_shapes(image_path, output_path="shapes_image.png"):
    image = cv2.imread(image_path)
    cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 3)
    cv2.circle(image, (300, 300), 50, (255, 0, 0), -1)
    cv2.imwrite(output_path, image)

def add_watermark(image_path, text, output_path="watermarked_image.png"):
    image = cv2.imread(image_path)
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(output_path, image)

def blend_images(image1_path, image2_path, alpha=0.5, output_path="blended_image.png"):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    cv2.imwrite(output_path, blended_image)

def dilate_image(image_path, output_path="dilated_image.png"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(output_path, dilated_image)

def erode_image(image_path, output_path="eroded_image.png"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite(output_path, eroded_image)

def corner_detection(image_path, output_path="corners_image.png"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    cv2.imwrite(output_path, image)

def median_filter(image_path, output_path="median_filtered_image.png"):
    image = cv2.imread(image_path)
    filtered_image = cv2.medianBlur(image, 5)
    cv2.imwrite(output_path, filtered_image)

def sketch_image(image_path, output_path="sketch_image.png"):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    cv2.imwrite(output_path, sketch)

def segment_image(image_path, k=3, output_path="segmented_image.png"):
    image = cv2.imread(image_path)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    cv2.imwrite(output_path, segmented_image)

def face_detection(image_path, output_path="faces_detected.png"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite(output_path, image)

# Example usage for each function
if __name__ == "__main__":
    grayscale("inputs/input_image.png", "outputs/gray_image.png")
    resize_image("inputs/input_image.png", 300, 300, "outputs/resized_image.png")
    rotate_90("inputs/input_image.png", "outputs/rotated_90_image.png")
    flip_horizontal("inputs/input_image.png", "outputs/flipped_image.png")
    increase_brightness("inputs/input_image.png", 100, "outputs/bright_image.png")
    gaussian_blur("inputs/input_image.png", "outputs/blurred_image.png")
    binary_threshold("inputs/input_image.png", "outputs/binary_image.png")
    histogram("inputs/input_image.png", "outputs/histogram.png")
    crop_image("inputs/input_image.png", 50, 50, 200, 200, "outputs/cropped_image.png")
    edge_detection("inputs/input_image.png", "outputs/edges.png")
    draw_shapes("inputs/input_image.png", "outputs/shapes_image.png")
    add_watermark("inputs/input_image.png", "Watermark Text", "outputs/watermarked_image.png")
    blend_images("inputs/input_image.png", "inputs/input_image2.png", 0.5, "outputs/blended_image.png")
    dilate_image("inputs/input_image.png", "outputs/dilated_image.png")
    erode_image("inputs/input_image.png", "outputs/eroded_image.png")
    corner_detection("inputs/input_image.png", "outputs/corners_image.png")
    median_filter("inputs/input_image.png", "outputs/median_filtered_image.png")
    sketch_image("inputs/input_image.png", "outputs/sketch_image.png")
    segment_image("inputs/input_image.png", 3, "outputs/segmented_image.png")
    


