import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict

from utils.helpers import sort_file, get_sequences, DATA_DIR, load_icebergs_by_frame, calculate_iou

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Tracking Evaluation and Ground Truth Matching Module

This module provides a complete pipeline for evaluating multi-object tracking (MOT) 
performance against ground truth annotations. It implements:

1. Ground Truth Matching: Filter tracking results to only include detections that 
   match ground truth objects using greedy IoU-based matching

2. Metric Computation: Calculate comprehensive tracking metrics including:
   - CLEAR metrics (CLR_Re, LocA, IDSW, Frag, MT, PT, ML)
   - Identity metrics (IDF1, IDR, IDP, IDTP, IDFN, IDFP)
   - Count metrics (Dets, GT_Dets, IDs, GT_IDs)

The evaluation follows MOTChallenge conventions and is compatible with TrackEval
for standardized benchmarking.

Typical Usage:
    >>> config = EvalConfig(dataset="hill/test", iou_threshold=0.3)
    >>> eval_tracking(config)

    Or step-by-step:
    >>> match_tracking_to_gt(config)  # Filter to GT coverage
    >>> calc_metrics(config)           # Compute all metrics
"""


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """
    Configuration for tracking evaluation against ground truth.

    This dataclass centralizes all parameters for the evaluation pipeline
    that filters tracking results to ground truth coverage using bidirectional IoU
    matching, then computes comprehensive tracking metrics.

    Configuration Categories:
        - Data: Dataset path and sequence selection
        - Matching: IoU threshold for detection-GT matching

    Attributes:
        dataset (str): Name/path of dataset to evaluate, relative to DATA_DIR
        iou_threshold (float): Minimum Intersection over Union for valid matches

    Workflow:
        1. Create config:
           >>> config = EvalConfig(dataset="hill/test", iou_threshold=0.3)

        2. Match tracking to GT (filters tracking results):
           >>> match_tracking_to_gt(config)
           # Creates: dataset/tracking/track_eval.txt

        3. Compute metrics:
           >>> metrics = calc_metrics(config)
           # Prints: Count, CLEAR, Identity metrics tables

        4. Or run complete pipeline:
           >>> eval_tracking(config)

    Output Metrics:
        Count:
            - Dets: Total detections in filtered tracking
            - GT_Dets: Total ground truth detections
            - IDs: Unique track IDs in filtered tracking
            - GT_IDs: Unique ground truth track IDs

        CLEAR:
            - CLR_Re: CLEAR Recall (TP / GT_Dets)
            - LocA: Localization Accuracy (average IoU of matches)
            - MTR/PTR/MLR: Mostly/Partially/Mostly Lost Track Ratios
            - CLR_TP/FN: True Positives / False Negatives
            - IDSW: ID switches (GT switches to different track ID)
            - Frag: Fragmentations (GT tracking interrupted)
            - MT/PT/ML: Mostly/Partially/Mostly Lost tracks (counts)

        Identity:
            - IDF1: ID F1-Score (harmonic mean of IDR and IDP)
            - IDR: ID Recall
            - IDP: ID Precision
            - IDTP/IDFN/IDFP: ID True/False Positives/Negatives

    Example:
        >>> # Standard evaluation with default IoU threshold
        >>> config = EvalConfig(dataset="hill/test")
        >>> eval_tracking(config)
    """
    # Data configuration
    dataset: str

    # Threshold configuration
    iou_threshold: float = 0.3


# ============================================================================
# METRIC CALCULATION
# ============================================================================

def calc_metrics(config: EvalConfig):
    """
    Calculate comprehensive tracking metrics after GT matching.

    This function loads ground truth and filtered tracking results, computes
    all standard MOT metrics, and displays them in TrackEval-style tables.
    Handles both single-sequence and multi-sequence datasets.

    Args:
        config (EvalConfig): Configuration with dataset path and IoU threshold

    Returns:
        dict: Nested dictionary of metrics

    Example:
        >>> config = EvalConfig(dataset="hill/test", iou_threshold=0.3)
        >>> metrics = calc_metrics(config)

        >>> # Access specific metrics
        >>> print(f"Overall IDF1: {metrics['COMBINED']['IDF1']:.3f}")
        >>> print(f"Recall: {metrics['COMBINED']['CLR_Re']:.3f}")
        >>> print(f"ID Switches: {metrics['COMBINED']['IDSW']}")

        >>> # Access sequence-specific metrics
        >>> for seq_name, seq_metrics in metrics.items():
        >>>     if seq_name != 'COMBINED':
        >>>         print(f"{seq_name}: IDF1={seq_metrics['IDF1']:.3f}")
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPUTING TRACKING METRICS")
    logger.info(f"{'=' * 80}\n")

    sequences = get_sequences(config.dataset)

    all_metrics = {}

    for sequence_name, paths in sequences.items():
        logger.info(f"Processing sequence: {sequence_name}")

        # Load data
        gt_by_frame = load_icebergs_by_frame(paths["ground_truth"])
        track_by_frame = load_icebergs_by_frame(paths["track_eval"])

        metrics = compute_sequence_metrics(gt_by_frame, track_by_frame, config.iou_threshold)
        all_metrics[sequence_name] = metrics

    # Compute combined metrics if multiple sequences
    if len(all_metrics) > 1:
        combined = combine_metrics(all_metrics)
        all_metrics["COMBINED"] = combined

    # Print all metrics in TrackEval style
    print_all_metrics(all_metrics)

    return all_metrics


def compute_sequence_metrics(gt_by_frame, track_by_frame, iou_threshold=0.3):
    """
    Compute all tracking metrics for a single sequence.

    This is the core metric computation function that implements the full
    MOTChallenge evaluation protocol for a single video sequence. It performs
    frame-by-frame matching and computes CLEAR, Identity, and Count metrics.

    Algorithm Overview:
        1. Frame-by-frame greedy IoU matching
        2. Count basic statistics (detections, IDs)
        3. Compute CLEAR metrics (recall, accuracy, ID switches, fragmentations)
        4. Compute track coverage (MT, PT, ML)
        5. Compute Identity metrics (IDF1, IDR, IDP)

    Args:
        gt_by_frame (dict): Ground truth organized by frame
        track_by_frame (dict): Tracking results organized by frame
        iou_threshold (float): Minimum IoU for valid match

    Returns:
        dict: All computed metrics for this sequence

    Matching Logic:
        For each GT in each frame:
            1. Find track with highest IoU
            2. If IoU ≥ threshold: Match found
            3. Otherwise: GT is missed (FN)

        Greedy: Each GT matched to at most one track per frame
                Each track matched to at most one GT per frame
    """
    frames = sorted(gt_by_frame.keys())

    # ========================================================================
    # 1. Frame-by-frame matching for CLEAR metrics
    # ========================================================================

    frame_matches = []  # List of (frame_id, gt_id, track_id, iou)
    gt_to_track = {}  # Dict[frame_id][gt_id] -> track_id
    track_to_gt = {}  # Dict[frame_id][track_id] -> gt_id

    CLR_TP = 0  # Matched detections
    CLR_FN = 0  # Missed GT

    all_ious = []  # Store all IoU values for LocA calculation

    for frame_id in frames:
        gts = gt_by_frame.get(frame_id, {})
        tracks = track_by_frame.get(frame_id, {})

        gt_to_track[frame_id] = {}
        track_to_gt[frame_id] = {}

        # Greedy matching: for each GT, find best track
        for gt_id, gt in gts.items():
            gt_bb = gt["bbox"]
            best_iou = 0.0
            best_track_id = None

            for track_id, track in tracks.items():
                track_bb = track["bbox"]
                # Convert box format from (x, y, w, h) to (xmin, ymin, xmax, ymax)
                box1 = [gt_bb[0], gt_bb[1], gt_bb[0] + gt_bb[2], gt_bb[1] + gt_bb[3]]
                box2 = [track_bb[0], track_bb[1], track_bb[0] + track_bb[2], track_bb[1] + track_bb[3]]
                iou = calculate_iou(box1, box2)

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= iou_threshold and best_track_id is not None:
                # Match found
                gt_to_track[frame_id][gt_id] = best_track_id
                track_to_gt[frame_id][best_track_id] = gt_id
                frame_matches.append((frame_id, gt_id, best_track_id, best_iou))
                all_ious.append(best_iou)  # Store IoU for LocA
                CLR_TP += 1
            else:
                # GT not matched
                CLR_FN += 1

    # ========================================================================
    # 2. Count metrics
    # ========================================================================

    # Total detections (after GT matching)
    Dets = sum(len(tracks) for tracks in track_by_frame.values())

    # Total GT detections
    GT_Dets = sum(len(gts) for gts in gt_by_frame.values())

    # Unique track IDs
    all_track_ids = set()
    for tracks in track_by_frame.values():
        all_track_ids.update(tracks.keys())
    IDs = len(all_track_ids)

    # Unique GT IDs
    all_gt_ids = set()
    for gts in gt_by_frame.values():
        all_gt_ids.update(gts.keys())
    GT_IDs = len(all_gt_ids)

    # ========================================================================
    # 3. CLEAR Recall and Localization Accuracy
    # ========================================================================

    CLR_Re = CLR_TP / GT_Dets if GT_Dets > 0 else 0.0

    # LocA: Average IoU of all matched pairs
    LocA = np.mean(all_ious) if len(all_ious) > 0 else 0.0

    # ========================================================================
    # 4. ID Switches (IDSW) and Fragmentations (Frag)
    # ========================================================================

    IDSW = 0
    Frag = 0

    # Track the last known track_id for each gt_id
    gt_last_track = {}  # gt_id -> last_track_id
    gt_last_frame = {}  # gt_id -> last_frame_id

    for frame_id in frames:
        for gt_id, track_id in gt_to_track.get(frame_id, {}).items():
            if gt_id in gt_last_track:
                # GT was tracked before
                if gt_last_track[gt_id] != track_id:
                    # ID switch: same GT, different track ID
                    IDSW += 1
                    gt_last_track[gt_id] = track_id

                # Check for fragmentation
                if int(frame_id) > int(gt_last_frame[gt_id]) + 1:
                    # There was a gap (GT not tracked in previous frame)
                    Frag += 1

            else:
                # First time tracking this GT
                gt_last_track[gt_id] = track_id

            gt_last_frame[gt_id] = int(frame_id)

    # ========================================================================
    # 5. Track Coverage (MT, PT, ML)
    # ========================================================================

    MT = 0  # Mostly Tracked (≥80% tracked)
    PT = 0  # Partially Tracked (20-80% tracked)
    ML = 0  # Mostly Lost (<20% tracked)

    for gt_id in all_gt_ids:
        # Count frames where this GT exists
        gt_frames = [f for f in frames if gt_id in gt_by_frame.get(f, {})]
        total_frames = len(gt_frames)

        if total_frames == 0:
            continue

        # Count frames where this GT was matched
        matched_frames = sum(1 for f in gt_frames if gt_id in gt_to_track.get(f, {}))

        coverage = matched_frames / total_frames

        if coverage >= 0.8:
            MT += 1
        elif coverage >= 0.2:
            PT += 1
        else:
            ML += 1

    # Coverage ratios
    MTR = MT / GT_IDs if GT_IDs > 0 else 0.0
    PTR = PT / GT_IDs if GT_IDs > 0 else 0.0
    MLR = ML / GT_IDs if GT_IDs > 0 else 0.0

    # ========================================================================
    # 6. Identity Metrics (IDTP, IDFN, IDFP)
    # ========================================================================

    # Build GT trajectories: gt_id -> list of (frame_id, track_id)
    gt_trajectories = defaultdict(list)
    for frame_id in frames:
        for gt_id, track_id in gt_to_track.get(frame_id, {}).items():
            gt_trajectories[gt_id].append((frame_id, track_id))

    # Build track trajectories: track_id -> list of (frame_id, gt_id)
    track_trajectories = defaultdict(list)
    for frame_id in frames:
        for track_id, gt_id in track_to_gt.get(frame_id, {}).items():
            track_trajectories[track_id].append((frame_id, gt_id))

    # Compute IDTP: longest common subsequence for each GT
    IDTP = 0
    for gt_id, gt_traj in gt_trajectories.items():
        # Find longest continuous match with any track
        track_segments = defaultdict(int)
        current_track = None
        current_length = 0

        for frame_id, track_id in gt_traj:
            if track_id == current_track:
                current_length += 1
            else:
                if current_track is not None:
                    track_segments[current_track] = max(track_segments[current_track], current_length)
                current_track = track_id
                current_length = 1

        # Don't forget last segment
        if current_track is not None:
            track_segments[current_track] = max(track_segments[current_track], current_length)

        # IDTP for this GT is longest continuous match
        if track_segments:
            IDTP += max(track_segments.values())

    # IDFN: GT frames not in longest matches
    IDFN = CLR_TP - IDTP

    # IDFP: Track frames not in longest matches
    # Similar calculation from track perspective
    IDTP_from_tracks = 0
    for track_id, track_traj in track_trajectories.items():
        gt_segments = defaultdict(int)
        current_gt = None
        current_length = 0

        for frame_id, gt_id in track_traj:
            if gt_id == current_gt:
                current_length += 1
            else:
                if current_gt is not None:
                    gt_segments[current_gt] = max(gt_segments[current_gt], current_length)
                current_gt = gt_id
                current_length = 1

        if current_gt is not None:
            gt_segments[current_gt] = max(gt_segments[current_gt], current_length)

        if gt_segments:
            IDTP_from_tracks += max(gt_segments.values())

    IDFP = CLR_TP - IDTP_from_tracks

    # Identity metrics
    IDR = IDTP / (IDTP + IDFN) if (IDTP + IDFN) > 0 else 0.0
    IDP = IDTP / (IDTP + IDFP) if (IDTP + IDFP) > 0 else 0.0
    IDF1 = 2 * IDTP / (2 * IDTP + IDFN + IDFP) if (2 * IDTP + IDFN + IDFP) > 0 else 0.0

    # ========================================================================
    # Return all metrics
    # ========================================================================

    return {
        # Count
        'Dets': Dets,
        'GT_Dets': GT_Dets,
        'IDs': IDs,
        'GT_IDs': GT_IDs,

        # CLEAR
        'CLR_Re': CLR_Re,
        'LocA': LocA,
        'CLR_TP': CLR_TP,
        'CLR_FN': CLR_FN,
        'IDSW': IDSW,
        'Frag': Frag,
        'MT': MT,
        'PT': PT,
        'ML': ML,
        'MTR': MTR,
        'PTR': PTR,
        'MLR': MLR,

        # Identity
        'IDF1': IDF1,
        'IDR': IDR,
        'IDP': IDP,
        'IDTP': IDTP,
        'IDFN': IDFN,
        'IDFP': IDFP,
    }


def combine_metrics(all_metrics):
    """
    Aggregate metrics across multiple sequences.

    This function combines per-sequence metrics into overall summary statistics
    for datasets containing multiple video sequences. It properly handles
    different aggregation strategies for different metric types.

    Aggregation Strategies:
        1. Summation: For count-based metrics (TP, FN, IDSW, etc.)
        2. Weighted Average: For ratio metrics (LocA)
        3. Recomputation: For derived metrics (CLR_Re, MTR, IDF1, etc.)

    Args:
        all_metrics (dict): Per-sequence metrics

    Returns:
        dict: Combined metrics across all sequences
    """
    combined = {}

    # Sum count metrics
    count_metrics = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs', 'CLR_TP', 'CLR_FN',
                     'IDSW', 'Frag', 'MT', 'PT', 'ML', 'IDTP', 'IDFN', 'IDFP']

    for metric in count_metrics:
        combined[metric] = sum(m[metric] for m in all_metrics.values())

    # Recompute derived metrics
    combined['CLR_Re'] = combined['CLR_TP'] / combined['GT_Dets'] if combined['GT_Dets'] > 0 else 0.0
    combined['MTR'] = combined['MT'] / combined['GT_IDs'] if combined['GT_IDs'] > 0 else 0.0
    combined['PTR'] = combined['PT'] / combined['GT_IDs'] if combined['GT_IDs'] > 0 else 0.0
    combined['MLR'] = combined['ML'] / combined['GT_IDs'] if combined['GT_IDs'] > 0 else 0.0

    # Average LocA across sequences (weighted by number of matches)
    total_matches = sum(m['CLR_TP'] for m in all_metrics.values())
    if total_matches > 0:
        combined['LocA'] = sum(m['LocA'] * m['CLR_TP'] for m in all_metrics.values()) / total_matches
    else:
        combined['LocA'] = 0.0

    combined['IDR'] = combined['IDTP'] / (combined['IDTP'] + combined[
        'IDFN']) if (combined['IDTP'] + combined['IDFN']) > 0 else 0.0
    combined['IDP'] = combined['IDTP'] / (combined['IDTP'] + combined[
        'IDFP']) if (combined['IDTP'] + combined['IDFP']) > 0 else 0.0
    combined['IDF1'] = 2 * combined['IDTP'] / (2 * combined['IDTP'] + combined['IDFN'] + combined['IDFP']) \
        if (2 * combined['IDTP'] + combined['IDFN'] + combined['IDFP']) > 0 else 0.0

    return combined


def print_all_metrics(all_metrics):
    """
    Print all metrics in TrackEval-style formatted tables.

    Displays evaluation results in a clean, professional format with separate
    tables for Count, CLEAR, Identity, and Derived metrics. Sequences are
    shown as rows, metrics as columns.

    Args:
        all_metrics (dict): Nested dictionary of metrics for all sequences
    """

    # ========================================================================
    # Count Metrics Table
    # ========================================================================

    logger.info(f"\n{'=' * 80}")
    logger.info("Count:")
    logger.info(f"{'=' * 80}")

    # Header
    logger.info(f"{'Sequence':<30} {'Dets':<12} {'GT_Dets':<12} {'IDs':<12} {'GT_IDs':<12}")
    logger.info(f"{'-' * 80}")

    # Rows
    for seq_name, metrics in all_metrics.items():
        logger.info(
            f"{seq_name:<30} "
            f"{metrics['Dets']:<12} "
            f"{metrics['GT_Dets']:<12} "
            f"{metrics['IDs']:<12} "
            f"{metrics['GT_IDs']:<12}"
        )

    # ========================================================================
    # CLEAR Metrics Table
    # ========================================================================

    logger.info(f"\n{'=' * 80}")
    logger.info("CLEAR:")
    logger.info(f"{'=' * 80}")

    # Header
    logger.info(
        f"{'Sequence':<30} "
        f"{'CLR_Re':<10} {'LocA':<10} {'MTR':<10} {'PTR':<10} {'MLR':<10} "
        f"{'CLR_TP':<10} {'CLR_FN':<10} {'IDSW':<10} {'Frag':<10} "
        f"{'MT':<8} {'PT':<8} {'ML':<8}"
    )
    logger.info(f"{'-' * 150}")

    # Rows
    for seq_name, metrics in all_metrics.items():
        logger.info(
            f"{seq_name:<30} "
            f"{metrics['CLR_Re']:<10.3f} "
            f"{metrics['LocA']:<10.3f} "
            f"{metrics['MTR']:<10.3f} "
            f"{metrics['PTR']:<10.3f} "
            f"{metrics['MLR']:<10.3f} "
            f"{metrics['CLR_TP']:<10} "
            f"{metrics['CLR_FN']:<10} "
            f"{metrics['IDSW']:<10} "
            f"{metrics['Frag']:<10} "
            f"{metrics['MT']:<8} "
            f"{metrics['PT']:<8} "
            f"{metrics['ML']:<8}"
        )

    # ========================================================================
    # Identity Metrics Table
    # ========================================================================

    logger.info(f"\n{'=' * 80}")
    logger.info("Identity:")
    logger.info(f"{'=' * 80}")

    # Header
    logger.info(
        f"{'Sequence':<30} "
        f"{'IDF1':<10} {'IDR':<10} {'IDP':<10} "
        f"{'IDTP':<10} {'IDFN':<10} {'IDFP':<10}"
    )
    logger.info(f"{'-' * 100}")

    # Rows
    for seq_name, metrics in all_metrics.items():
        logger.info(
            f"{seq_name:<30} "
            f"{metrics['IDF1']:<10.3f} "
            f"{metrics['IDR']:<10.3f} "
            f"{metrics['IDP']:<10.3f} "
            f"{metrics['IDTP']:<10} "
            f"{metrics['IDFN']:<10} "
            f"{metrics['IDFP']:<10}"
        )

    # ========================================================================
    # Derived Metrics Summary
    # ========================================================================

    logger.info(f"\n{'=' * 80}")
    logger.info("Derived Metrics:")
    logger.info(f"{'=' * 80}")

    # Header
    logger.info(
        f"{'Sequence':<30} "
        f"{'ID_Ratio':<12} {'IDSW/track':<12} {'Frag/track':<12}"
    )
    logger.info(f"{'-' * 80}")

    # Rows
    for seq_name, metrics in all_metrics.items():
        id_ratio = metrics['IDs'] / metrics['GT_IDs'] if metrics['GT_IDs'] > 0 else 0.0
        idsw_per_track = metrics['IDSW'] / metrics['GT_IDs'] if metrics['GT_IDs'] > 0 else 0.0
        frag_per_track = metrics['Frag'] / metrics['GT_IDs'] if metrics['GT_IDs'] > 0 else 0.0

        logger.info(
            f"{seq_name:<30} "
            f"{id_ratio:<12.2f} "
            f"{idsw_per_track:<12.2f} "
            f"{frag_per_track:<12.2f}"
        )

    logger.info(f"\n{'=' * 80}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def eval_tracking(config: EvalConfig):
    """
    Run complete end-to-end tracking evaluation pipeline.

    This is the main entry point for evaluating tracking results. It performs
    both steps of the evaluation process: (1) matching tracking to ground truth,
    and (2) computing comprehensive metrics.

    Pipeline Steps:
        1. match_tracking_to_gt(config):
           - Load ground truth and tracking results
           - Perform frame-by-frame greedy IoU matching
           - Filter tracking to only GT-matched detections
           - Save filtered results to track_eval.txt

        2. calc_metrics(config):
           - Load ground truth and filtered tracking
           - Compute Count, CLEAR, and Identity metrics
           - Display results in formatted tables
           - Return metrics dictionary

    Args:
        config (EvalConfig): Evaluation configuration
            Required fields:
                - dataset: Path to dataset (contains ground_truth/ and tracking/)
                - iou_threshold: IoU threshold for matching (default: 0.3)

    Returns:
        None (metrics are printed to console)
        Call calc_metrics() directly if you need the metrics dictionary

        Contains only tracking detections that match ground truth above IoU threshold

    Example Usage:
        >>> # Basic evaluation with default threshold
        >>> config = EvalConfig(dataset="hill/test")
        >>> eval_tracking(config)

        >>> # Evaluation with custom IoU threshold
        >>> config = EvalConfig(dataset="hill/test", iou_threshold=0.5)
        >>> eval_tracking(config)


    Performance:
        - Typical dataset (1000 frames, 50 objects/frame):
          - Matching: ~5-10 seconds
          - Metrics: ~1-2 seconds
          - Total: ~10-15 seconds

    Notes:
        - Compatible with TrackEval and MOTChallenge conventions
    """
    # Step 1: Match tracking to GT
    match_tracking_to_gt(config)

    # Step 2: Calculate metrics
    calc_metrics(config)


def match_tracking_to_gt(config: EvalConfig):
    """
    Match tracking results to ground truth using greedy IoU matching.

    This function filters tracking results to only include detections that
    match ground truth objects above an IoU threshold. It performs frame-by-frame
    greedy matching and saves filtered results for subsequent metric computation.

    Algorithm:
        For each frame:
            For each GT object:
                1. Find tracking detection with highest IoU
                2. If IoU ≥ threshold: Keep this tracking detection
                3. Otherwise: GT is unmatched (will count as FN in metrics)

        Save all matched tracking detections to track_eval.txt

    Args:
        config (EvalConfig): Configuration with dataset path and IoU threshold
            Required:
                config.dataset: Dataset path (contains ground_truth/ and tracking/)
                config.iou_threshold: Minimum IoU for valid match (default: 0.3)

    Raises:
        FileNotFoundError: If ground_truth/ or tracking/ directory missing
        ValueError: If data files malformed or empty

    Example:
        >>> config = EvalConfig(dataset="hill/test", iou_threshold=0.3)
        >>> match_tracking_to_gt(config)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FILTER TRACKING RESULTS TO GROUND TRUTH\n")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"IoU threshold: {config.iou_threshold}\n")

    # Load all sequences in the dataset
    sequences = get_sequences(config.dataset)
    logger.info(f"Found {len(sequences)} sequence(s)")
    logger.info(f"Sequences: {', '.join(sequences.keys())}\n")

    # Process each sequence independently
    for sequence_name, paths in sequences.items():
        logger.info(f"Processing sequence: {sequence_name}")

        gt_by_frame = load_icebergs_by_frame(paths["ground_truth"])
        track_by_frame = load_icebergs_by_frame(paths["tracking"])
        frames = sorted(gt_by_frame.keys())

        gt_matches = []
        # Process each frame
        for frame_id in frames:
            gts = gt_by_frame.get(frame_id, {})
            tracks = track_by_frame.get(frame_id, {})
            frame_gt_matches = []

            for gt in gts.values():
                gt_bb = gt["bbox"]
                best_match = 0.0
                candidate = None

                for track in tracks.values():
                    track_bb = track["bbox"]
                    box1 = [gt_bb[0], gt_bb[1], gt_bb[0] + gt_bb[2], gt_bb[1] + gt_bb[3]]
                    box2 = [track_bb[0], track_bb[1], track_bb[0] + track_bb[2], track_bb[1] + track_bb[3]]
                    iou = calculate_iou(box1, box2)
                    if iou > best_match and iou > config.iou_threshold:
                        best_match = iou
                        candidate = track

                if candidate:
                    if candidate not in frame_gt_matches:
                        candidate["frame_id"] = int(frame_id)
                        frame_gt_matches.append(candidate)

            gt_matches.append(frame_gt_matches)

        # Write to file in MOTChallenge format
        with open(paths["track_eval"], 'w') as f:
            for frame_matches in gt_matches:
                for match in frame_matches:
                    x, y, w, h = match['bbox']
                    f.write(
                        f"{match['frame_id']},{match['id']},"
                        f"{x},{y},{w},{h},"
                        f"{match['conf']},1,-1,-1\n"
                    )

        # Sort by frame number for easier analysis
        sort_file(paths["track_eval"])

        logger.info(f"✓ Filtered tracking saved to: {paths['track_eval']}")
        logger.info("")  # Empty line for readability