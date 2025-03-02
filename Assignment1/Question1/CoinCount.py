import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load image
image = cv2.imread(r"C:\Users\DELL\Desktop\VR\Assignment1\Question1\Input_Images\Img_3.jpg")

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for metallic colors (includes both dark & bright coins)
lower_bound = np.array([0, 0, 30])   # Looser lower bound for dark & reflective coins
upper_bound = np.array([180, 150, 255])  # Extended saturation to include reflections

# Create a mask for metallic regions
hsv_mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Convert filtered image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 5)

# Apply Otsu's Thresholding
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Edge Detection (Canny)
edges = cv2.Canny(blurred, 100, 200)

# Combine HSV mask and edge mask
combined_mask = cv2.bitwise_and(binary, hsv_mask)

# Morphological Closing to remove small gaps
kernel = np.ones((5, 5), np.uint8)
processed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and circularity
filtered_contours = []
for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    if perimeter == 0:
        continue
    
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    # Accept contours with reasonable size and circular shape
    if area > 500 and 0.6 < circularity < 1.3:  # Adjusted circularity range
        filtered_contours.append(c)

# Draw detected coins
image_copy = image.copy()
cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), 2)

# Create and save the combined output image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show Segmentation Output
axes[0].imshow(processed, cmap='gray')
axes[0].set_title("Binary Segmentation Output")
axes[0].axis("off")

# Show Coin Count Output
axes[1].imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Coin Count Output - {len(filtered_contours)} Coins")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(r"C:\Users\DELL\Desktop\VR\Assignment1\Question1\Output_Images\Output_Img_3.jpg")  # Save combined image
plt.show()

# Print number of detected coins
print(f"Number of coins detected: {len(filtered_contours)}")
