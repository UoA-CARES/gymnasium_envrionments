import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

def draw_contour(game_frame):
    # Optional: Apply color thresholding to focus on dark areas (if your game frame is in color)
    # _, black_areas = cv2.threshold(game_frame, 15, 255, cv2.THRESH_BINARY_INV)
    # For grayscale images, you might skip this step or adjust it based on the game's color scheme

    # Apply Canny Edge Detection
    edges = cv2.Canny(game_frame, threshold1=400, threshold2=700)

    # Use dilation to consolidate edges and make them easier to detect as contours
    kernel = np.ones((5,5), np.uint8)  # Adjust kernel size as needed
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the dilated image
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw contours on the original game frame for visualization
    # contour_frame = cv2.cvtColor(game_frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR for coloring contours
    cv2.drawContours(game_frame, contours, -1, (0, 0, 0), thickness=cv2.FILLED)  # Draw contours in green
    
    # Create a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(game_frame)

    # Draw the contours on the mask with white color and filled (-1)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Invert the mask, so the area outside the contours is white and inside is black
    mask_inv = cv2.bitwise_not(mask)

    # Combine the inverted mask with the original image, so areas outside the contours are set to white
    result = cv2.bitwise_or(game_frame, mask_inv)

    return result

def get_contour(game_frame):
    # Optional: Apply color thresholding to focus on dark areas (if your game frame is in color)
    # _, black_areas = cv2.threshold(game_frame, 15, 255, cv2.THRESH_BINARY_INV)
    # For grayscale images, you might skip this step or adjust it based on the game's color scheme

    # Apply Canny Edge Detection
    edges = cv2.Canny(game_frame, threshold1=500, threshold2=700)

    # Use dilation to consolidate edges and make them easier to detect as contours
    kernel = np.ones((5,5), np.uint8)  # Adjust kernel size as needed
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the dilated image
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw contours on the original game frame for visualization
    # contour_frame = cv2.cvtColor(game_frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR for coloring contours
    cv2.drawContours(game_frame, contours, -1, (0, 255, 0), 3)  # Draw contours in green

    return contours
