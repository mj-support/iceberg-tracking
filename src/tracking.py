import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from preprocessing import get_tiles_with_overlap
from utils.paths import DATA_DIR


def merge_tiles(dataset):
    """
    Merges tiles based on overlapping regions and updates detection IDs in the provided dataset.

    :param overlap: The overlap ratio for merging tiles (default is 0.025).
    :param run: The identifier for the current run (default is "01").
    """
    print("Merge tiles...")
    tiles = get_tiles_with_overlap(dataset)  # Fetch tiles with overlap information
    det_tiles_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
    det_merged_file = os.path.join(DATA_DIR, dataset, "detections", "det_raw.txt")
    existing_ids = {}  # To store existing IDs for image name

    with open(det_tiles_file, "r") as infile, open(det_merged_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split(",")  # Split the line by comma

            if len(parts) < 10:  # Skip invalid lines
                continue

            image_name = parts[0]  # Extract image name
            tile_suffix = image_name[-1]  # Last character indicates tile
            original_id = int(parts[1])  # Original detection ID

            if tile_suffix in tiles:
                x_offset = tiles[tile_suffix]["xmin"]
                y_offset = tiles[tile_suffix]["ymin"]

                # Adjust bounding box coordinates based on tile offset
                x = float(parts[2]) + x_offset
                y = float(parts[3]) + y_offset
                width = float(parts[4])
                height = float(parts[5])

                clean_image_name = image_name[:-2]  # Remove tile suffix for clean image name

                if clean_image_name not in existing_ids:
                    existing_ids[clean_image_name] = set()

                # Ensure unique ID across frames for the same object
                if original_id in existing_ids[clean_image_name]:
                    new_id = 1
                    while new_id in existing_ids[clean_image_name]:
                        new_id += 1
                else:
                    new_id = original_id

                existing_ids[clean_image_name].add(new_id)

                # Write the updated line with the new ID and bounding box values
                new_line = f"{clean_image_name},{new_id},{x},{y},{width},{height},{parts[6]},{parts[7]},{parts[8]},{parts[9]}\n"
                outfile.write(new_line)


def iou(bb1, bb2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    :param bb1: First bounding box as (x, y, width, height).
    :param bb2: Second bounding box as (x, y, width, height).
    :return: IoU value as float between 0 and 1.
    """
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bb1_area = w1 * h1
    bb2_area = w2 * h2
    union_area = bb1_area + bb2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def perspective_weighted_distance(bb1, bb2, k=3, img_height=4000):
    """
    Calculate the perspective-weighted distance between two bounding boxes.

    :param bb1: First bounding box as (x, y, width, height).
    :param bb2: Second bounding box as (x, y, width, height).
    :param k: Scaling factor for the perspective (higher = stronger correction).
    :param img_height: Height of the image, used for scaling the perspective.
    :return: Adjusted Euclidean distance between bounding boxes, scaled by perspective.
    """
    x1, y1, _, _ = bb1
    x2, y2, _, _ = bb2

    d_euklid = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Compute the midpoint of the Y-coordinates of the bounding boxes
    y_mittel = (y1 + y2) / 2

    # Apply perspective scaling based on the Y-midpoint
    scale_factor = 1 + k * (1 - (y_mittel / img_height))

    return d_euklid * scale_factor


def euclidean_distance(bb1, bb2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.

    :param bb1: First bounding box as (x, y, width, height).
    :param bb2: Second bounding box as (x, y, width, height).
    :return: Euclidean distance as float.
    """
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    return np.sqrt((x1 + w1 / 2 - x2 - w2 / 2) ** 2 + (y1 + h1 / 2 - y2 - h2 / 2) ** 2)


def match_detections(prev_boxes, new_boxes, threshold=0.01):
    """
    Assign new bounding boxes to previous detections by computing a cost matrix.

    :param prev_boxes: List of previous bounding boxes.
    :param new_boxes: List of new bounding boxes.
    :param threshold: Threshold for IoU to consider as a match.
    :return: Dictionary of matches and a set of unmatched new boxes.
    """
    cost_matrix = np.zeros((len(prev_boxes), len(new_boxes)))

    # Calculate cost based on IoU and perspective distance
    for i, prev_bb in enumerate(prev_boxes):
        for j, new_bb in enumerate(new_boxes):
            iou_score = iou(prev_bb, new_bb)
            distance = perspective_weighted_distance(prev_bb, new_bb)
            cost_matrix[i, j] = -iou_score + 0.01 * distance

    # Use Hungarian Algorithm to find the best assignments with minimal cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    unmatched_new = set(range(len(new_boxes)))

    for r, c in zip(row_ind, col_ind):
        if iou(prev_boxes[r], new_boxes[c]) > threshold or perspective_weighted_distance(prev_boxes[r],
                                                                                         new_boxes[c]) < 50: #### 150 für hill!!!
            matches[c] = r
            unmatched_new.discard(c)

    return matches, unmatched_new


def process_tracking(dataset, max_inactive_frames=8):
    """
    Load detection data from file and apply tracking based on matching bounding boxes across frames.

    :param detection_file: File containing detection information.
    :param output_file: File to store the tracking results.
    :param max_inactive_frames: Maximum number of frames an object can be inactive before being discarded.
    """
    print("Tracking...")
    prev_objects = []  # List of (ID, Bounding Box)
    inactive_objects = {}  # Dictionary {ID: (Bounding Box, Frames Inactive)}
    next_id = 1  # Starting ID for new objects
    det_merged_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
    tracking_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

    with open(det_merged_file, 'r') as f:
        lines = f.readlines()

    results = []  # List for storing updated results with new IDs

    frame_data = {}
    for line in lines:
        parts = line.strip().split(',')
        frame_id = parts[0]
        bbox = list(map(float, parts[2:6]))  # Extract bounding box coordinates
        if frame_id not in frame_data:
            frame_data[frame_id] = []
        frame_data[frame_id].append((bbox, parts))

    # Process frames in order
    for frame_id in sorted(frame_data.keys()):
        new_boxes = [entry[0] for entry in frame_data[frame_id]]
        matches, unmatched_new = match_detections([b for _, b in prev_objects], new_boxes)

        new_objects = []
        matched_ids = set()

        # Assign IDs to new boxes or match with existing ones
        for new_idx, new_box in enumerate(new_boxes):
            if new_idx in matches:
                obj_id = prev_objects[matches[new_idx]][0]
                matched_ids.add(obj_id)
            else:
                obj_id = None
                for inactive_id, (old_box, inactive_count) in list(inactive_objects.items()):
                    if iou(old_box, new_box) > 0.3:
                        obj_id = inactive_id
                        del inactive_objects[inactive_id]
                        break
                if obj_id is None:
                    obj_id = next_id
                    next_id += 1
            new_objects.append((obj_id, new_box))

        new_inactive_objects = {}
        for obj_id, bbox in prev_objects:
            if obj_id not in matched_ids:
                inactive_count = inactive_objects.get(obj_id, (None, 0))[1] + 1
                if inactive_count <= max_inactive_frames:
                    new_inactive_objects[obj_id] = (bbox, inactive_count)

        inactive_objects = new_inactive_objects
        prev_objects = new_objects  # Remember current objects for next frame
        for i, (bbox, parts) in enumerate(frame_data[frame_id]):
            parts[1] = str(new_objects[i][0])  # Update ID
            results.append(','.join(parts) + '\n')

    # Write tracking results to output file
    with open(tracking_file, 'w') as f:
        f.writelines(results)


def filter_tracking_output(dataset, min_count=0):
    """
    Filters tracking output to remove objects that occur fewer than a minimum number of times.

    :param output_file: File containing the tracking results.
    :param min_count: Minimum count for an ID to be considered valid.
    """
    print("Filter tracking output...")
    tracking_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

    det_data = pd.read_csv(tracking_file, header=None)
    det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'x', 'y',
                        'z']

    id_counts = det_data['ID'].value_counts()
    valid_ids = set(id_counts[id_counts >= min_count].index)  # IDs that appear at least `min_count` times

    filtered_data = det_data[det_data['ID'].isin(valid_ids)]

    filtered_data.to_csv(tracking_file, index=False, header=False)


def filter_nested_icebergs(dataset, threshold=0.6):
    """
    Remove icebergs that are at least 'threshold' percent inside another iceberg in the same frame.

    :param output_file: Path to tracking output file.
    :param threshold: IoU threshold to consider an iceberg nested inside another.
    """

    def intersection_area(box1, box2):
        """
        Calculate the intersection area of two bounding boxes.

        :param box1: A tuple (x, y, width, height) representing the first bounding box.
        :param box2: A tuple (x, y, width, height) representing the second bounding box.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the coordinates of the intersection box
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        # Calculate intersection width and height, return area
        inter_width = max(0, xb - xa)
        inter_height = max(0, yb - ya)
        return inter_width * inter_height

    def is_mostly_inside(box1, box2, threshold):
        """
        Check if box1 is at least 'threshold' percent inside box2.

        :param box1: A tuple (x, y, width, height) representing the first bounding box.
        :param box2: A tuple (x, y, width, height) representing the second bounding box.
        :param threshold (float): The minimum percentage of box1 that should be inside box2.
        """
        area1 = box1[2] * box1[3]  # width * height of box1
        inter_area = intersection_area(box1, box2)  # intersection area with box2
        # Check if the intersection area is large enough relative to box1's area
        return (inter_area / area1) >= threshold if area1 > 0 else False

    tracking_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

    # Open the output file and read the lines
    with open(tracking_file, 'r') as f:
        lines = f.readlines()

    frame_data = {}

    # Read data and organize it by frame
    for line in lines:
        parts = line.strip().split(',')
        frame_id = parts[0]  # Frame ID
        bbox = list(map(float, parts[2:6]))  # x, y, width, height of bounding box

        # Group bounding boxes by frame
        if frame_id not in frame_data:
            frame_data[frame_id] = []
        frame_data[frame_id].append((bbox, parts))

    filtered_results = []

    # Iterate through frames and filter out nested bounding boxes
    for frame_id in sorted(frame_data.keys()):
        objects = frame_data[frame_id]

        # Filter out bounding boxes mostly inside others
        filtered_objects = []
        for i, (box, parts) in enumerate(objects):
            if not any(
                    is_mostly_inside(box, other_box, threshold) for j, (other_box, _) in enumerate(objects) if i != j):
                filtered_objects.append(parts)

        # Append valid detections (those not nested inside others)
        for parts in filtered_objects:
            filtered_results.append(','.join(parts) + '\n')

    # Write the filtered results back to the output file
    with open(tracking_file, 'w') as f:
        f.writelines(filtered_results)


def visualize_detections(dataset, file, save_images, start_index=0):
    """
    Visualize the tracking results by drawing bounding boxes on images and displaying them.

    :param img_folder: The folder containing the images to be processed.
    :param tracking_file: The path to the txt-file containing tracking data for the detections.
    :param save_images: Whether or not to save the images with bounding boxes drawn.
    :param start_index: The index of the first image to process. Default is 0 (process from the beginning).
    """
    print("Visualize detections...")
    image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
    colormap = {}  # Store colors for each object ID

    # Load the tracking data
    det_data = pd.read_csv(file, header=None)
    det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'x', 'y', 'z']

    # Sort and process all images in the folder
    sorted_images = sorted(os.listdir(image_dir))

    for i, img_name in enumerate(sorted_images):
        if i < start_index:
            continue  # Skip images before the start index

        if img_name.endswith(".jpg") or img_name.endswith(".JPG"):
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


def main():
    dataset = "fjord_2min_2023-08"
    image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
    det_tiles_file = os.path.join(DATA_DIR, dataset, "detections", "det_tiles.txt")
    det_merged_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
    tracking_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

    #merge_tiles(dataset=dataset)
    #visualize_detections(dataset, det_merged_file , save_images=False, start_index=0)
    #process_tracking(dataset)
    #filter_tracking_output(dataset, min_count=4)
    #filter_nested_icebergs(dataset, threshold=0.8)
    visualize_detections(dataset, tracking_file, save_images=False, start_index=200)

    #os.system("ffmpeg -framerate 2 -pattern_type glob -i 'results/*.JPG' -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p results/tracking.mp4 -y")


if __name__ == '__main__':
    main()