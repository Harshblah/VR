import numpy as np
import cv2
import imutils
import os
import matplotlib.pyplot as plt

# Define the folder where unstitched images are stored
image_directory = r"C:\Users\DELL\Desktop\VR\Assignment1\Question2\InputImages"

# Collect all image files with supported formats
image_files = sorted([f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_list = []

# Read images and store them
for img_file in image_files:
    image_path = os.path.join(image_directory, img_file)
    current_image = cv2.imread(image_path)
    if current_image is not None:
        image_list.append(current_image)

# Ensure we have at least two images for stitching
if len(image_list) < 2:
    print("Error: Not enough images loaded for stitching. Check image folder path and contents.")
    exit()

# Ensure output directory exists
output_dir = r"C:\Users\DELL\Desktop\VR\Assignment1\Question2\ProcessedOutputs"
os.makedirs(output_dir, exist_ok=True)

# ▶️ Show and Save the Original Images
fig, axs = plt.subplots(1, len(image_list), figsize=(16, 4))
for i, img in enumerate(image_list):
    axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f"Image {i+1}")
    axs[i].axis("off")
plt.suptitle("Original Images for Stitching", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "original_images.png"))
plt.show()

# Initialize ORB feature detector
orb_detector = cv2.ORB_create()

# Detect keypoints and store images with keypoints
keypoints_images = []
for image in image_list:
    keypoints_detected = orb_detector.detect(image, None)
    keypoint_overlay = cv2.drawKeypoints(image, keypoints_detected, None, color=(0, 0, 255))
    keypoints_images.append(keypoint_overlay)

# ▶️ Show and Save ORB Keypoints Detected Images
fig, axs = plt.subplots(1, len(keypoints_images), figsize=(16, 4))
for i, img in enumerate(keypoints_images):
    axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f"Keypoints in Image {i+1}")
    axs[i].axis("off")
plt.suptitle("Detected ORB Keypoints", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "keypoints_images.png"))
plt.show()

# Create an OpenCV Stitcher instance
stitcher = cv2.Stitcher_create()
status, stitched_output = stitcher.stitch(image_list)

# If stitching is successful
if status == cv2.Stitcher_OK:
    resized_stitched = cv2.resize(stitched_output, (800, 600))

    # ▶️ Show and Save Resized Stitched Image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(resized_stitched, cv2.COLOR_BGR2RGB))
    plt.title("Resized Stitched Image")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "stitched_resized.png"))
    plt.show()

    # Convert to grayscale
    stitched_output = cv2.copyMakeBorder(stitched_output, 5, 5, 5, 5, cv2.BORDER_CONSTANT, (0, 0, 0))
    grayscale_img = cv2.cvtColor(stitched_output, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY)[1]
    threah_cpy = thresh_img.copy()

    contours = cv2.findContours(threah_cpy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x,y,w,h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x+w,y+h),255,-1)

    min_rect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect,None)
        sub = cv2.subtract(min_rect, thresh_img)

    contours = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    areaOI = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(areaOI)

    stitched_output = stitched_output[y:y+h,x:x+w]
    final_output = stitched_output
    # grayscale_img = cv2.cvtColor(stitched_output, cv2.COLOR_BGR2GRAY)

    # # Threshold the image to create a mask of non-black pixels
    # _, binary_mask = cv2.threshold(grayscale_img, 1, 255, cv2.THRESH_BINARY)

    # # Find contours in the binary mask
    # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     # Get the bounding rectangle of the largest contour
    #     x, y, w, h = cv2.boundingRect(cv2.convexHull(np.vstack(contours)))

    #     # Crop the image using the bounding box
    #     final_output = stitched_output[y:y + h, x:x + w]
    # else:
    #     print("Warning: No valid contours found. Using full image as fallback.")
    #     final_output = stitched_output  # Fallback

    # Save and display the final cleaned image
    final_output_path = os.path.join(output_dir, "stitched_cropped_clean.png")
    cv2.imwrite(final_output_path, final_output)

    # ▶️ Show and Save Final Cropped Stitched Image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Image without Black Borders")
    plt.axis("off")
    plt.savefig(final_output_path)
    plt.show()
else:
    print(f"Image stitching failed with status code: {status}. Please check the input images.")
