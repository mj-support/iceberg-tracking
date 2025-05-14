import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.paths import DATA_DIR


def visualize(dataset, file, image_format="jpg", save_images=False, start_index=0):
    """
    Visualize the results by drawing bounding boxes on images and displaying them.

    Args:
        dataset (str): Name of the dataset directory
        image_format (str): File formats of the images (file extension without the dot)
        file (str): The path to the txt-file containing data of the images and icebergs.
        save_images (bool): Whether or not to save the images with bounding boxes drawn.
        start_index: The index of the first image to process. Default is 0 (process from the beginning).
    """
    print("Visualize detections...")
    image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
    image_format = f".{image_format}".lower()
    colormap = {}  # Store colors for each object ID

    # Load the tracking data
    det_data = pd.read_csv(file, header=None)
    det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'x', 'y', 'z']

    # Sort and process all images in the folder
    sorted_images = sorted(os.listdir(image_dir))

    for i, img_name in enumerate(sorted_images):
        if i < start_index:
            continue  # Skip images before the start index

        if img_name.lower().endswith(image_format):
            image_path = f"{image_dir}/{img_name}"
            img = cv2.imread(image_path)
            image_no = img_name[:-4]  # Get the frame ID from image name

            if img is None:
                print(f"Image {image_path} not found.")
                return

            # Filter detections for the current frame
            frame_data = det_data[det_data['frame'] == image_no]

            # Draw bounding boxes and labels on the image
            for _, row in frame_data.iterrows():
                object_id = int(row['ID'])

                # Assign a random color to each object ID (if not already assigned)
                if object_id not in colormap:
                    np.random.seed(object_id)  # Ensure consistent color assignment
                    color = tuple(map(int, np.random.randint(0, 255, size=3)))
                    colormap[object_id] = color
                color = colormap[object_id]

                # Extract bounding box coordinates
                bbox_left, bbox_top = int(row['bbox_left']), int(row['bbox_top'])
                bbox_width, bbox_height = int(row['bbox_width']), int(row['bbox_height'])

                # Draw the bounding box and label on the image
                cv2.rectangle(img, (bbox_left, bbox_top), (bbox_left + bbox_width, bbox_top + bbox_height), color, 2)
                cv2.putText(img, str(object_id), (bbox_left, bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display the image
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(img_name)
            plt.axis('off')
            plt.show()

            # Save the image if required
            if save_images:
                output_path = os.path.join(DATA_DIR, dataset, "results", "images")
                cv2.imwrite(output_path, img)
                print(f"Saved: {output_path}")