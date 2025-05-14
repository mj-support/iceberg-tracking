import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from utils.paths import DATA_DIR
from utils.visualize import visualize


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IcebergTracker:
    """
    A class for tracking icebergs across image frames based on detection data.
    """

    def __init__(self, dataset, image_format="jpg"):
        """
        Initialize the IcebergTracker with dataset and configuration parameters.

        Args:
            dataset (str): Name of the dataset directory
            image_format (str): File formats of the images (file extension without the dot)
        """
        self.dataset = dataset
        self.image_format = f".{image_format}".lower()

        # Set up file paths
        self.image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
        self.detections_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
        self.tracking_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)

        # Set tracking parameters
        # Threshold to preselect candidates
        self.cost_threshold = 0.3
        self.distance_top_k = None
        self.size_threshold = None
        self.iou_threshold = None
        # Weight for the cost matrix
        self.distance_weight = None
        self.size_weight = None
        self.iou_weight = None
        # Additional parameters
        self.max_inactive_frames = None
        self.perspective_k = None
        # Postprocessing filter
        self.min_detection_count = None
        self.nested_threshold = None

        self.image_height = None

        # Internal tracking state
        self.next_id = 1
        self.image_data = {}
        self.results = []

        logger.info(f"Initialized IcebergTracker for dataset: {dataset}")

    def track(self, cost_threshold=0.3, distance_top_k=0.1, size_threshold=0.3, iou_threshold=0.0,
              distance_weight=1.0, size_weight=1.0, iou_weight=1.0, max_inactive_frames=8, perspective_k=3,
              min_detection_count=4, nested_threshold=0.8,):
        """
        Perform tracking of icebergs across frames with configurable parameters.

        Args:
            cost_threshold: Score of the cost matrix needs to be lower than this threshold
            distance_top_k: Consider only the top X% closest new icebergs
            size_threshold: Minimum size similarity for a valid match
            iou_threshold: Minimum IoU to consider a valid match
            distance_weight: Weight for distance in cost calculation
            size_weight: Weight for size similarity in cost calculation
            iou_weight: Weight for IoU in cost calculation
            max_inactive_frames: Maximum number of frames an iceberg can be inactive before being discarded
            perspective_k: Perspective correction factor for distance calculation
            min_detection_count: Minimum number of detections required to keep an iceberg track
            nested_threshold: Threshold for filtering nested icebergs (0.8 means 80% overlap)
        """
        logger.info("Starting tracking process...")
        # Threshold to preselect candidates
        self.cost_threshold = cost_threshold
        self.distance_top_k = distance_top_k
        self.size_threshold = size_threshold
        self.iou_threshold = iou_threshold
        # Weight for the cost matrix
        self.distance_weight = distance_weight
        self.size_weight = size_weight
        self.iou_weight = iou_weight
        # Additional parameters
        self.max_inactive_frames = max_inactive_frames
        self.perspective_k = perspective_k
        # Postprocessing filter
        self.min_detection_count = min_detection_count
        self.nested_threshold = nested_threshold

        # Load detections
        self._load_detections()

        # Initialize tracking state
        prev_icebergs = []  # List of (ID, Bounding Box)
        inactive_icebergs = {}  # Dictionary {ID: (Bounding Box, Frames Inactive)}
        self.results = []  # List for storing updated results with new IDs

        for image_id in self.image_data:
            new_boxes = [entry[0] for entry in self.image_data[image_id]]
            prev_boxes = [b for _, b in prev_icebergs]

            # Match new detections with active icebergs
            matches, unmatched_new = self._match_detections(prev_boxes, new_boxes)

            new_icebergs, inactive_icebergs = self._process_detections(matches, unmatched_new, prev_icebergs, inactive_icebergs, new_boxes)

            # Update tracking results with assigned IDs
            for i, (bbox, parts) in enumerate(self.image_data[image_id]):
                # Find the corresponding new iceberg index
                for obj_id, obj_bbox in new_icebergs:
                    if obj_bbox == bbox:  # Find matching box
                        parts[1] = str(obj_id)  # Update ID
                        break
                self.results.append(','.join(parts) + '\n')

            prev_icebergs = new_icebergs

        # Write tracking results to file
        with open(self.tracking_file, 'w') as f:
            f.writelines(self.results)

        logger.info(f"Tracking completed. Results written to {self.tracking_file}")

        # Apply post-processing filters
        self._filter_tracking_output()
        self._filter_nested_icebergs()

    def _load_detections(self):
        """
        Load detection data from file and organize by frame.

        Returns:
            dict: Dictionary with frame IDs as keys and lists of detections as values
        """
        logger.info(f"Loading detections from {self.detections_file}")

        random_image = [f for f in os.listdir(self.image_dir) if f.lower().endswith(self.image_format)][0]
        with Image.open(os.path.join(self.image_dir, random_image)) as img:
            self.image_height = img.height

        try:
            with open(self.detections_file, 'r') as f:
                lines = f.readlines()

            image_data = {}
            for line in lines:
                parts = line.strip().split(',')
                image_id = parts[0]
                bbox = list(map(float, parts[2:6]))  # Extract bounding box coordinates

                if image_id not in image_data:
                    image_data[image_id] = []

                image_data[image_id].append((bbox, parts))

            self.image_data = image_data
            logger.info(f"Loaded {len(lines)} detections across {len(image_data)} frames")
            return image_data

        except Exception as e:
            logger.error(f"Error loading detections: {e}")
            raise

    def _match_detections(self, prev_boxes, new_boxes):
        """
        Match new detections to previous ones using a preselection approach
        followed by cost matrix optimization.

        Args:
            prev_boxes: List of previous bounding boxes
            new_boxes: List of new bounding boxes

        Returns:
            tuple: (matches, unmatched_new) where matches is a dict mapping
                  new box index to prev box index, and unmatched_new is a set
                  of indices of new boxes that couldn't be matched
        """
        if not prev_boxes or not new_boxes:
            return {}, set(range(len(new_boxes)))

        # Step 1: Preselect candidates based on thresholds
        candidates = self._preselect_candidates(prev_boxes, new_boxes)

        # Step 2: Calculate cost matrix for valid candidates only
        cost_matrix = self._calculate_cost_matrix(prev_boxes, new_boxes, candidates)

        # Step 3: Use Hungarian Algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = {}
        unmatched_new = set(range(len(new_boxes)))

        # Process matches, excluding those with high cost (non-candidates)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e6:  # Only include valid matches (not high cost)
                matches[c] = r
                unmatched_new.discard(c)

        return matches, unmatched_new

    def _preselect_candidates(self, prev_boxes, new_boxes):
        """
        Preselect candidate matches based on thresholds before calculating costs.

        For each previous box, identifies new boxes that meet all threshold criteria:
        - Within the top k% closest based on distance
        - Above the IoU threshold
        - Above the size similarity threshold

        Args:
            prev_boxes: List of previous bounding boxes
            new_boxes: List of new bounding boxes

        Returns:
            List of tuples (prev_idx, new_idx) for all valid candidate matches
        """
        candidates = []

        # Early exit if no boxes
        if not prev_boxes or not new_boxes:
            return candidates

        # Calculate all distances first
        distance_matrix = np.zeros((len(prev_boxes), len(new_boxes)))
        for i, prev_bb in enumerate(prev_boxes):
            for j, new_bb in enumerate(new_boxes):
                distance_matrix[i, j] = self._calculate_perspective_weighted_distance(prev_bb, new_bb)

        # For each previous box, determine candidates
        for i, prev_bb in enumerate(prev_boxes):
            # Get distances from this previous box to all new boxes
            distances_from_prev = distance_matrix[i, :]

            # Calculate how many matches to keep based on distance_top_k
            keep_count = max(1, int(len(new_boxes) * self.distance_top_k))

            # Find the distance threshold that keeps only the top k% closest
            if len(new_boxes) > 0:
                sorted_distances = np.sort(distances_from_prev)
                distance_threshold = sorted_distances[min(keep_count, len(sorted_distances) - 1)]
            else:
                continue  # No new boxes to match

            for j, new_bb in enumerate(new_boxes):
                # Check all threshold criteria
                distance = distance_matrix[i, j]
                iou_score = self._calculate_iou(prev_bb, new_bb)
                size_similarity = self._calculate_size_similarity(prev_bb, new_bb)

                if (distance <= distance_threshold and
                        iou_score >= self.iou_threshold and
                        size_similarity >= self.size_threshold):
                    candidates.append((i, j))

        return candidates

    def _calculate_cost_matrix(self, prev_boxes, new_boxes, candidates):
        """
        Calculate a cost matrix for matching detections based on multiple metrics.
        Only calculates costs for preselected candidate pairs.

        Args:
            prev_boxes: List of previous bounding boxes
            new_boxes: List of new bounding boxes
            candidates: List of tuples (prev_idx, new_idx) of preselected candidates

        Returns:
            numpy.ndarray: Cost matrix with shape (len(prev_boxes), len(new_boxes))
        """
        cost_matrix = np.full((len(prev_boxes), len(new_boxes)), 1e6)  # Default to high cost

        if not candidates:
            return cost_matrix

        # Calculate maximum distance among valid candidates for normalization
        max_distance = 1.0  # Default
        for i, j in candidates:
            distance = self._calculate_perspective_weighted_distance(prev_boxes[i], new_boxes[j])
            max_distance = max(max_distance, distance)

        # Calculate costs only for valid candidates
        for i, j in candidates:
            prev_bb = prev_boxes[i]
            new_bb = new_boxes[j]

            # Calculate metrics
            iou_score = self._calculate_iou(prev_bb, new_bb)
            distance = self._calculate_perspective_weighted_distance(prev_bb, new_bb)
            size_similarity = self._calculate_size_similarity(prev_bb, new_bb)

            # Calculate normalized costs
            iou_cost = 1 - iou_score
            normalized_distance = distance / max_distance
            size_diff_cost = 1 - size_similarity

            # Combine costs with weights
            cost_matrix[i, j] = ((self.iou_weight * iou_cost +
                                 self.distance_weight * normalized_distance +
                                 self.size_weight * size_diff_cost) /
                                 (self.iou_weight + self.distance_weight + self.size_weight))

            # Check if score is below cost_threshold
            if cost_matrix[i, j] > self.cost_threshold:
                cost_matrix[i, j] = 1e6

        return cost_matrix

    def _calculate_perspective_weighted_distance(self, bb1, bb2):
        """
        Calculate the perspective-weighted distance between two bounding boxes.
        Objects farther away (higher in the image) get a higher distance penalty.

        Args:
            bb1: First bounding box as (x, y, width, height)
            bb2: Second bounding box as (x, y, width, height)

        Returns:
            float: Perspective-weighted distance
        """
        x1, y1, _, _ = bb1
        x2, y2, _, _ = bb2

        # Calculate basic Euclidean distance
        d_euclid = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Compute the midpoint of the Y-coordinates (higher y means lower in image)
        y_midpoint = (y1 + y2) / 2

        # Apply perspective scaling based on the Y-midpoint
        # Objects higher in the image (smaller y) get a higher distance penalty
        scale_factor = 1 + self.perspective_k * (1 - (y_midpoint / self.image_height))

        return d_euclid * scale_factor

    def _calculate_iou(self, bb1, bb2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bb1: First bounding box as (x, y, width, height)
            bb2: Second bounding box as (x, y, width, height)

        Returns:
            float: IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        # Calculate intersection coordinates
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        # Calculate intersection area
        inter_width = max(0, xb - xa)
        inter_height = max(0, yb - ya)
        inter_area = inter_width * inter_height

        # Calculate box1 area
        area1 = w1 * h1

        # Check if the intersection area is large enough relative to box1's area
        return (inter_area / area1)


        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        # Calculate intersection coordinates
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Calculate areas
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bb1_area = w1 * h1
        bb2_area = w2 * h2
        union_area = bb1_area + bb2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _calculate_size_similarity(self, bb1, bb2):
        """
        Calculate similarity in size between two bounding boxes.

        Args:
            bb1: First bounding box as (x, y, width, height)
            bb2: Second bounding box as (x, y, width, height)

        Returns:
            float: Size similarity score between 0 and 1 (1 means identical size)
        """
        _, _, w1, h1 = bb1
        _, _, w2, h2 = bb2

        area1 = w1 * h1
        area2 = w2 * h2

        # Avoid division by zero
        if area1 == 0 and area2 == 0:
            return 1.0
        elif area1 == 0 or area2 == 0:
            return 0.0

        # Calculate size similarity as ratio of smaller to larger area
        return min(area1, area2) / max(area1, area2)

    def _process_detections(self, matches, unmatched_new, prev_icebergs, inactive_icebergs, new_boxes):
        """
        Process detection matches to maintain iceberg tracking across frames.

        This method handles the core tracking logic by:
        1. Updating matched active icebergs with their new positions
        2. Reactivating inactive icebergs when they match with new detections
        3. Creating new iceberg IDs for previously undetected objects
        4. Managing the inactive icebergs list (objects not detected in current frame)

        Args:
        matches: Dictionary mapping indices of new detections to indices of previous icebergs
        unmatched_new: Set of indices for new detections that didn't match any previous icebergs
        prev_icebergs: List of (iceberg_id, bounding_box) for icebergs from previous frame
        inactive_icebergs: Dictionary mapping iceberg IDs to (bounding_box, inactive_count) for icebergs
            that weren't detected in recent frames but are still being tracked
        new_boxes: List of bounding boxes for all detections in current frame

        Returns:
            tuple: (new_icebergs, new_inactive_icebergs) where:
                - new_icebergs is a list of (iceberg_id, bounding_box) for current frame
                - new_inactive_icebergs is updated dictionary of inactive icebergs
        """
        # Initialize new iceberg list for this frame
        new_icebergs = []
        # Starting ID for new objects
        self.next_id = 1

        # Step 1: Process matches with active icebergs
        matched_ids = set()
        for new_idx, prev_idx in matches.items():
            obj_id = prev_icebergs[prev_idx][0]
            matched_ids.add(obj_id)
            new_icebergs.append((obj_id, new_boxes[new_idx]))

        # Step 2: Try to match remaining unmatched detections with inactive icebergs
        if unmatched_new and inactive_icebergs:
            # Extract boxes from inactive icebergs
            inactive_boxes = [box for box, _ in inactive_icebergs.values()]
            inactive_ids = list(inactive_icebergs.keys())

            # Get unmatched new boxes and create a mapping to original indices
            unmatched_new_list = list(unmatched_new)
            unmatched_new_boxes = [new_boxes[idx] for idx in unmatched_new_list]

            # Match unmatched detections with inactive icebergs
            inactive_matches, still_unmatched = self._match_detections(inactive_boxes, unmatched_new_boxes)

            # Process matches with inactive icebergs
            matched_unmatched = set()  # Track which unmatched_new indices we've processed
            for new_idx, inactive_idx in inactive_matches.items():
                if new_idx < len(unmatched_new_list):  # Ensure index is valid
                    obj_id = inactive_ids[inactive_idx]
                    original_new_idx = unmatched_new_list[new_idx]

                    # Reactivate this inactive iceberg
                    del inactive_icebergs[obj_id]
                    new_icebergs.append((obj_id, new_boxes[original_new_idx]))
                    matched_unmatched.add(original_new_idx)

            # Update unmatched_new to remove the ones we just matched
            unmatched_new = unmatched_new - matched_unmatched

        # Step 3: Create new IDs for any remaining unmatched detections
        for new_idx in unmatched_new:
            obj_id = self.next_id
            self.next_id += 1
            new_icebergs.append((obj_id, new_boxes[new_idx]))

        # Step 4: Update inactive objects list
        new_inactive_icebergs = {}
        for obj_id, bbox in prev_icebergs:
            if obj_id not in matched_ids:
                inactive_count = inactive_icebergs.get(obj_id, (None, 0))[1] + 1
                if inactive_count <= self.max_inactive_frames:
                    new_inactive_icebergs[obj_id] = (bbox, inactive_count)

        # Add previously inactive objects that remain inactive
        for obj_id, (bbox, count) in inactive_icebergs.items():
            if obj_id not in matched_ids and obj_id not in new_inactive_icebergs:
                count += 1
                if count <= self.max_inactive_frames:
                    new_inactive_icebergs[obj_id] = (bbox, count)

        return new_icebergs, new_inactive_icebergs

    def _filter_tracking_output(self):
        """
        Filter tracking output to remove objects that appear less than min_count times.

        Args:
            min_count: Minimum number of appearances required to keep an iceberg track
        """
        logger.info(f"Filtering tracks with less than {self.min_detection_count} detections")

        try:
            # Read tracking data
            det_data = pd.read_csv(self.tracking_file, header=None)
            det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width',
                                'bbox_height', 'confidence', 'x', 'y', 'z']

            # Count occurrences of each ID
            id_counts = det_data['ID'].value_counts()
            valid_ids = set(id_counts[id_counts >= self.min_detection_count].index)

            # Filter by valid IDs
            filtered_data = det_data[det_data['ID'].isin(valid_ids)]

            # Save filtered data
            filtered_data.to_csv(self.tracking_file, index=False, header=False)

            logger.info(f"Filtered out {len(id_counts) - len(valid_ids)} tracks with fewer than {self.min_detection_count} detections")

        except Exception as e:
            logger.error(f"Error filtering tracking output: {e}")
            raise

    def _filter_nested_icebergs(self):
        """
        Remove icebergs that are at least 'threshold' percent inside another iceberg in the same frame.
        """
        logger.info(f"Filtering nested icebergs with threshold {self.nested_threshold}")

        try:
            # Read the tracking file to get current results
            with open(self.tracking_file, 'r') as f:
                lines = f.readlines()

            # Organize data by frame
            frame_data = {}
            for line in lines:
                parts = line.strip().split(',')
                frame_id = parts[0]  # Frame ID
                bbox = list(map(float, parts[2:6]))  # x, y, width, height

                if frame_id not in frame_data:
                    frame_data[frame_id] = []

                frame_data[frame_id].append((bbox, parts))

            filtered_results = []

            # Process each frame
            for frame_id in sorted(frame_data.keys()):
                objects = frame_data[frame_id]

                # Filter out nested bounding boxes
                filtered_objects = []
                for i, (box, parts) in enumerate(objects):
                    # Check if this box is mostly inside any other box
                    is_nested = False
                    for j, (other_box, _) in enumerate(objects):
                        iou_score = self._calculate_iou(box, other_box)
                        if i != j and iou_score >= self.nested_threshold:
                            is_nested = True
                            break

                    if not is_nested:
                        filtered_objects.append(parts)

                # Add non-nested objects to results
                for parts in filtered_objects:
                    filtered_results.append(','.join(parts) + '\n')

            # Write filtered results
            with open(self.tracking_file, 'w') as f:
                f.writelines(filtered_results)

            logger.info(
                f"Nested iceberg filtering completed. Removed {len(lines) - len(filtered_results)} nested icebergs.")

        except Exception as e:
            logger.error(f"Error filtering nested icebergs: {e}")
            raise

    def calculate_euclidean_distance(self, bb1, bb2):
        """
        Calculate the Euclidean distance between the centers of two bounding boxes.

        Args:
            bb1: First bounding box as (x, y, width, height)
            bb2: Second bounding box as (x, y, width, height)

        Returns:
            float: Euclidean distance
        """
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        # Calculate center points
        center1_x, center1_y = x1 + w1 / 2, y1 + h1 / 2
        center2_x, center2_y = x2 + w2 / 2, y2 + h2 / 2

        return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


def main():
    dataset = "fjord_2min_2023-08"
    tracker = IcebergTracker(dataset, image_format="JPG")

    # Run tracking with default parameters
    tracker.track(
        cost_threshold=0.3,
        distance_top_k=0.1,
        size_threshold=0.3,
        iou_threshold=0.03,
        distance_weight=1.0,
        size_weight=1.0,
        iou_weight=1.0,
        max_inactive_frames=2,
        perspective_k=3,
        min_detection_count=10,
        nested_threshold=0.8,
    )

    visualize(dataset, tracker.tracking_file, save_images=False, start_index=200)



if __name__ == "__main__":
    main()