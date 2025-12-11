import pandas as pd
import logging
from dataclasses import dataclass

from utils.helpers import sort_file, get_sequences, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Tracking Evaluation and Ground Truth Matching Module

This module provides functionality for evaluating iceberg tracking results against
ground truth annotations. It implements bidirectional IoU-based matching to create
filtered tracking outputs that can be evaluated using standard MOT metrics.
The results can be used for evaluation with TrackEval.
"""


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """
    Configuration for tracking evaluation against ground truth.

    This dataclass centralizes all parameters for the evaluation preparation pipeline
    that filters tracking results to ground truth coverage using bidirectional IoU
    matching. The filtered results can then be evaluated using TrackEval or similar
    MOT evaluation tools.

    Configuration Categories:
        - Data: Dataset path and sequence selection
        - Matching: IoU threshold for detection-GT matching

    Attributes:
        dataset (str): Name/path of dataset to evaluate
            Examples: "hill/test", "columbia/ice_melange"
            Must contain ground_truth/, detections/, and tracking/ directories

        iou_threshold (float): Minimum Intersection over Union for valid matches. Default: 0.5

    Workflow:
        1. Create config: config = EvalConfig(dataset="hill/test", iou_threshold=0.5)
        2. Run evaluation: filter_tracking_to_gt(config)
        3. Results saved to: dataset/tracking/track_eval.txt
        4. Evaluate with TrackEval for metrics (HOTA, MOTA, IDF1)

    Example:
        >>> # Standard evaluation
        >>> config = EvalConfig(dataset="hill/test")
        >>> filter_tracking_to_gt(config)
    """
    # Data configuration
    dataset: str

    # Threshold configuration
    iou_threshold: float = 0.5


# ============================================================================
# IoU COMPUTATION
# ============================================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    IoU is a standard metric for measuring bounding box overlap. It ranges
    from 0 (no overlap) to 1 (perfect alignment) and is widely used in
    object detection and tracking for matching and evaluation.

    Args:
        box1 (list or tuple): First bounding box [x, y, width, height]
            - x, y: Top-left corner coordinates
            - width, height: Box dimensions
        box2 (list or tuple): Second bounding box [x, y, width, height]

    Returns:
        float: IoU score in range [0.0, 1.0]
            - 0.0 if boxes don't overlap
            - 1.0 if boxes are identical
            - >0.0 indicates some overlap
    """
    # Convert box format from (x, y, w, h) to (xmin, ymin, xmax, ymax)
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection rectangle boundaries
    # Intersection top-left: maximum of both top-lefts
    # Intersection bottom-right: minimum of both bottom-rights
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    # Check if boxes actually overlap
    # If right edge is left of left edge, or bottom is above top, no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    # Union = Area1 + Area2 - Intersection
    area1 = box1[2] * box1[3]  # width × height
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    # Return IoU, protecting against division by zero
    return intersection / union if union > 0 else 0.0


# ============================================================================
# BIDIRECTIONAL MATCHING
# ============================================================================

def match_detections_to_gt(paths, iou_threshold):
    """
    Match detections to ground truth using bidirectional IoU matching.

    This function implements a rigorous bidirectional matching strategy where
    a detection-GT pair is only accepted if they are mutually each other's
    best match. This ensures high-quality correspondences and reduces false
    positive matches.

    Args:
        paths (dict): Dictionary of file paths from get_sequences()
            Required keys: 'ground_truth', 'detections'
        iou_threshold (float): Minimum IoU for a valid match
            Typical values:
            - 0.3: Very permissive (low quality)
            - 0.5: Standard (MOT Challenge default)
            - 0.7: Strict (high quality)

    Returns:
        list or None: List of matched detection dictionaries, each containing:
            - 'frame': Frame number (int)
            - 'det_id': Original detection ID (int)
            - 'x', 'y', 'w', 'h': Bounding box (float)
            - 'conf': Detection confidence (float)
            - 'x_3d', 'y_3d', 'z_3d': 3D coordinates (int)
            - 'gt_id': Ground truth ID for reference (int)

            Returns None if required files are missing
    """
    # Define column names for CSV files (MOTChallenge format)
    gt_cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_3d', 'y_3d', 'z_3d']

    # Validate that required files exist
    if not paths["ground_truth"].exists():
        logger.warning(f"No gt.txt found at {paths['ground_truth']}, skipping...")
        return None
    if not paths["detections"].exists():
        logger.warning(f"No detections found at {paths['detections']}, skipping...")
        return None

    # Load ground truth and detection data
    gt_df = pd.read_csv(paths["ground_truth"], header=None, names=gt_cols)
    detections_df = pd.read_csv(paths["detections"], header=None, names=gt_cols)

    # Initialize tracking structures
    matched_detections = []  # List of matched detection dictionaries
    matched_gt_indices = set()  # Track which GT entries were matched

    # Group by frame for efficient processing
    gt_by_frame = gt_df.groupby('frame')
    detections_by_frame = detections_df.groupby('frame')

    # Process each frame independently
    for frame_num in gt_df['frame'].unique():
        # Skip frames with no detections
        if frame_num not in detections_by_frame.groups:
            continue

        # Get GT and detections for this frame
        gt_frame = gt_by_frame.get_group(frame_num)
        det_frame = detections_by_frame.get_group(frame_num)

        # Phase 1: Forward matching (GT → Detection)
        # For each GT, find its best matching detection
        gt_to_det = {}  # gt_idx -> (best_det_idx, best_iou)

        for gt_idx, gt_row in gt_frame.iterrows():
            # Extract GT bounding box
            gt_box = [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']]

            # Find detection with highest IoU
            best_iou = 0.0
            best_det_idx = None

            for det_idx, det_row in det_frame.iterrows():
                det_box = [det_row['x'], det_row['y'], det_row['w'], det_row['h']]
                iou = calculate_iou(gt_box, det_box)

                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            # Record match if IoU meets threshold
            if best_iou >= iou_threshold and best_det_idx is not None:
                gt_to_det[gt_idx] = (best_det_idx, best_iou)

        # Phase 2: Backward matching (Detection → GT)
        # For each detection, find its best matching GT
        det_to_gt = {}  # det_idx -> (best_gt_idx, best_iou)

        for det_idx, det_row in det_frame.iterrows():
            # Extract detection bounding box
            det_box = [det_row['x'], det_row['y'], det_row['w'], det_row['h']]

            # Find GT with highest IoU
            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, gt_row in gt_frame.iterrows():
                gt_box = [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']]
                iou = calculate_iou(gt_box, det_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Record match if IoU meets threshold
            if best_iou >= iou_threshold and best_gt_idx is not None:
                det_to_gt[det_idx] = (best_gt_idx, best_iou)

        # Phase 3: Bidirectional matching (mutual agreement)
        # Only keep matches where both GT and detection prefer each other
        for gt_idx, (best_det_idx, forward_iou) in gt_to_det.items():
            # Check if this detection also chose this GT
            if best_det_idx in det_to_gt:
                best_gt_idx, backward_iou = det_to_gt[best_det_idx]

                # Verify mutual agreement
                if best_gt_idx == gt_idx:
                    # Valid bidirectional match!
                    gt_row = gt_frame.loc[gt_idx]
                    det_row = det_frame.loc[best_det_idx]

                    # Store the matched detection with full information
                    matched_detections.append({
                        'frame': det_row['frame'],
                        'det_id': det_row['id'],  # Original detection ID
                        'x': det_row['x'],
                        'y': det_row['y'],
                        'w': det_row['w'],
                        'h': det_row['h'],
                        'conf': det_row['conf'],
                        'x_3d': det_row['x_3d'],
                        'y_3d': det_row['y_3d'],
                        'z_3d': det_row['z_3d'],
                        'gt_id': gt_row['id']  # Keep GT ID for reference
                    })
                    matched_gt_indices.add(gt_idx)

    # Report unmatched ground truth entries
    unmatched_gt = gt_df.loc[~gt_df.index.isin(matched_gt_indices)]

    logger.info("\n===== UNMATCHED GT ENTRIES =====")
    if unmatched_gt.empty:
        logger.info("All GT entries matched ✓")
    else:
        logger.info(f"Unmatched GT count: {len(unmatched_gt)}")
        logger.info("\n" + str(unmatched_gt[['frame', 'id', 'x', 'y', 'w', 'h']]))
    logger.info("================================\n")

    # Print matching statistics
    logger.info(f"Total detections: {len(detections_df)}")
    logger.info(f"Matched detections to GT: {len(matched_detections)}")
    logger.info(f"GT entries: {len(gt_df)}")

    # Compute match rate
    match_rate = len(matched_detections) / len(gt_df) * 100 if len(gt_df) > 0 else 0
    logger.info(f"Match rate: {match_rate:.1f}%")

    return matched_detections


def update_with_track_ids(matched_detections, paths):
    """
    Update matched detections with track IDs from tracking results.

    This function takes detections that were successfully matched to ground truth
    and assigns them track IDs from the tracking output. If a detection was not
    tracked (missed by tracker), it gets assigned a new unique ID.

    Args:
        matched_detections (list): List of matched detection dicts from
            match_detections_to_gt(), each containing:
            - 'frame', 'det_id', 'x', 'y', 'w', 'h', 'conf'
            - 'x_3d', 'y_3d', 'z_3d', 'gt_id'
        paths (dict): Dictionary of file paths from get_sequences()
            Required key: 'tracking'

    Returns:
        list or None: List of track dictionaries ready for evaluation, each containing:
            - 'frame': Frame number (int)
            - 'id': Track ID (int) - either from tracker or newly assigned
            - 'x', 'y', 'w', 'h': Bounding box (float)
            - 'conf': Confidence score (float)
            - 'x_3d', 'y_3d', 'z_3d': 3D coordinates (int)

            Returns None if tracking file is missing
    """
    # Define column names for CSV files (MOTChallenge format)
    gt_cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_3d', 'y_3d', 'z_3d']

    # Validate that tracking file exists
    if not paths["tracking"].exists():
        logger.warning(f"No tracking file found at {paths['tracking']}, skipping...")
        return None

    # Load tracking results
    tracking_df = pd.read_csv(paths["tracking"], header=None, names=gt_cols)

    # Find maximum track ID to ensure new IDs don't conflict
    max_track_id = tracking_df['id'].max()
    next_id = max_track_id + 1

    # Build lookup dictionary: (frame, bbox) -> track_id
    # This enables O(1) lookup to find track IDs
    tracking_lookup = {}
    for _, row in tracking_df.iterrows():
        frame = row['frame']
        # Use tuple of floats as key (immutable, hashable)
        bbox = (float(row['x']), float(row['y']), float(row['w']), float(row['h']))
        track_id = row['id']
        tracking_lookup[(frame, bbox)] = track_id

    # Update matched detections with track IDs
    matched_tracks = []
    untracked_count = 0

    for det in matched_detections:
        frame = det['frame']
        bbox = (float(det['x']), float(det['y']), float(det['w']), float(det['h']))

        # Look up track ID using (frame, bbox) key
        if (frame, bbox) in tracking_lookup:
            # Detection was successfully tracked - use tracking ID
            track_id = tracking_lookup[(frame, bbox)]
        else:
            # Detection was missed by tracker - assign new unique ID
            track_id = next_id
            next_id += 1
            untracked_count += 1

        # Create final track entry for evaluation
        matched_tracks.append({
            'frame': frame,
            'id': track_id,  # Updated with track ID (existing or new)
            'x': det['x'],
            'y': det['y'],
            'w': det['w'],
            'h': det['h'],
            'conf': det['conf'],
            'x_3d': det['x_3d'],
            'y_3d': det['y_3d'],
            'z_3d': det['z_3d']
        })

    # Print statistics
    logger.info(f"Matched detections with existing tracks: {len(matched_detections) - untracked_count}")
    logger.info(f"Matched detections assigned new IDs: {untracked_count}")
    logger.info(f"Total matched tracks: {len(matched_tracks)}")

    # Compute tracking recall
    if len(matched_detections) > 0:
        tracking_recall = (len(matched_detections) - untracked_count) / len(matched_detections) * 100
        logger.info(f"Tracking recall: {tracking_recall:.1f}%")

    return matched_tracks


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def filter_tracking_to_gt(config: EvalConfig):
    """
    Complete pipeline to filter tracking results to ground truth coverage.

    This is the main entry point for the evaluation preparation pipeline. It
    orchestrates the complete workflow from loading data to saving filtered
    results ready for MOT evaluation. This function prepares data for evaluation
    but doesn't compute metrics. Use TrackEval or similar tools for actual
    metric computation.

    Pipeline Steps:
        1. Load dataset sequences
        2. For each sequence:
           a. Match detections to ground truth (bidirectional IoU)
           b. Find tracked detections and assign track IDs
           c. Save filtered tracking file
        3. Report statistics

    Args:
        config (EvalConfig): Complete eval configuration

    Next Steps After Filtering:
        1. Verify output files exist: tracking/track_eval.txt
        2. Run TrackEval for metrics:
        3. Analyze results:
           - HOTA: Overall tracking quality
           - MOTA: Detection + association accuracy
           - IDF1: ID consistency
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Filtering Tracking Results to Ground Truth")
    logger.info(f"{'=' * 60}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"IoU threshold: {config.iou_threshold}")
    logger.info(f"{'=' * 60}\n")

    # Load all sequences in the dataset
    sequences = get_sequences(config.dataset)
    logger.info(f"Found {len(sequences)} sequence(s)")
    logger.info(f"Sequences: {', '.join(sequences.keys())}\n")

    # Process each sequence independently
    for sequence_name, paths in sequences.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing sequence: {sequence_name}")
        logger.info(f"{'=' * 60}\n")

        # Step 1: Match detections to ground truth using bidirectional IoU
        logger.info("Step 1: Matching detections to ground truth...")
        matched_detections = match_detections_to_gt(paths, iou_threshold=config.iou_threshold)

        # Skip if matching failed or no matches found
        if matched_detections is None or not matched_detections:
            logger.warning("No matches found, skipping sequence.\n")
            continue

        # Step 2: Update matched detections with track IDs from tracking output
        logger.info("\nStep 2: Assigning track IDs to matched detections...")
        matched_tracks = update_with_track_ids(matched_detections, paths)

        # Skip if track ID assignment failed
        if not matched_tracks:
            logger.warning("No tracked detections found, skipping sequence.\n")
            continue

        # Step 3: Save filtered tracking results
        logger.info("\nStep 3: Saving filtered tracking results...")

        # Convert to DataFrame for easy manipulation
        filtered_df = pd.DataFrame(matched_tracks)

        # Sort by frame and ID for consistent ordering
        filtered_df = filtered_df.sort_values(['frame', 'id'])

        # Ensure correct data types for MOTChallenge format
        for col in ['frame', 'id', 'x_3d', 'y_3d', 'z_3d']:
            filtered_df[col] = filtered_df[col].astype(int)

        # Save to CSV (no header for MOTChallenge format)
        filtered_df.to_csv(paths["track_eval"], header=False, index=False)

        # Sort file by frame and ID (redundant but ensures correctness)
        sort_file(paths["track_eval"])

        logger.info(f"✓ Filtered tracking saved to: {paths['track_eval']}")
        logger.info(f"  Total entries: {len(filtered_df)}")
        logger.info(f"  Unique tracks: {filtered_df['id'].nunique()}")
        logger.info(f"  Frame range: {filtered_df['frame'].min()} - {filtered_df['frame'].max()}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Pipeline complete!")
    logger.info(f"{'=' * 60}\n")
    logger.info("Next steps:")
    logger.info("1. Verify track_eval.txt files were created")
    logger.info("2. Run TrackEval to compute metrics")
    logger.info("3. Analyze HOTA, MOTA, IDF1 scores")