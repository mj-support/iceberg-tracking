import logging
import numpy as np
import os
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

from utils.feature_extraction import get_distance, get_size_similarity, get_appearance_similarity, \
    min_max_normalize, get_score, get_gt_thresholds
from utils.helpers import load_icebergs_by_frame, sort_file, get_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)


"""
Iceberg Multi-Object Tracking with Kalman Filtering and Spatial Indexing

This module implements a multi-object tracking (MOT) system specifically
designed for tracking icebergs across timelapse image sequences. It combines classical
tracking techniques with modern deep learning features.

Key Features:
    1. Kalman Filtering: Motion prediction using constant velocity model
    2. Spatial Indexing: Grid-based hashing for O(1) candidate selection
    3. Appearance-Based Matching: Deep learning embeddings
    4. Multi-Feature Fusion: Combines appearance, distance, and size
    5. Track Management: Lifecycle handling with age/confidence

Architecture:
    IcebergTrackingConfig: Configuration dataclass with all hyperparameters
    IcebergTrack: Individual track with Kalman filter and history
    SpatialIndex: Grid-based spatial hashing structure
    IcebergTracker: Main tracking orchestrator

Tracking Pipeline (per frame):
    1. Predict: Use Kalman filter to predict new positions
    2. Index: Build spatial index of detections for fast lookup
    3. Match: Find best track-detection correspondences with Bidirectional Matching as default
    4. Update: Update matched tracks with new detections
    5. Delete: Remove tracks that are too old (max_age exceeded)
    6. Create: Initialize new tracks for unmatched detections
    7. Output: Export confirmed tracks to results
"""


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IcebergTrackingConfig:
    """
    Configuration for iceberg tracking with motion prediction.

    This dataclass centralizes all hyperparameters for the tracking system,
    making it easy to experiment with different configurations and ensuring
    reproducibility.

    Configuration Categories:
        - Data: Dataset paths and processing limits
        - Algorithm: Kalman filtering and matching strategy
        - Thresholds: Similarity requirements for matching
        - Weights: Relative importance of features
        - Track Management: Lifecycle parameters
        - Kalman Filter: Motion model parameters

    Attributes:
        # Data Configuration
        dataset (str): Name/path of dataset to process
        seq_start_index (int): Starting frame index (for testing)
        seq_length_limit (int | None): Max frames to process (None = all)

        # Algorithm Configuration
        use_kalman (bool): Enable Kalman filtering for motion prediction
        use_spatial_index (bool): Enable grid-based spatial indexing
        bidirectional_matching (bool): Matching algorithm selection
            - True: Mutual best match (conservative, fewer false matches)
            - False: Global greedy (permissive, better continuity)

        # Threshold Configuration (similarity requirements)
        thresholds (dict): Minimum similarity values for matching
            - 'appearance': Min appearance similarity [0, 1]
            - 'distance': Max spatial distance in pixels
            - 'size': Min size similarity [0, 1]
        threshold_tolerance (float): Factor make the threshold less strict*
        get_gt_thresholds (bool): Get thresholds from ground truth data
        gt_thresholds (str): Path to training / ground truth directory

        # Weight Configuration (relative importance)
        weight_appearance (float): Weight for appearance similarity
        weight_euclidean_distance (float): Weight for euclidean distance similarity
        weight_kalman_distance (float): Weight for kalman distance similarity
        weight_size (float): Weight for size similarity

        # Track Management Configuration
        max_age (int): Max frames a track can be unmatched before deletion
        min_iceberg_id_count (int): Min detections required to keep a track
        min_iceberg_size (float): Min bounding box area for new tracks

        # Kalman Filter Parameters
        process_noise (float): Motion model uncertainty (pixels)
        measurement_noise (float): Detection uncertainty (pixels)

    Example Configurations:
        >>> config = IcebergTrackingConfig(
        ...     dataset="hill/train",
        ...     use_kalman=True,
        ...     thresholds={'appearance': 0.25, 'distance': 200, 'size': 0.25},
        ... )
    """
    # Data configuration
    dataset: str
    seq_start_index: int = 0
    seq_length_limit: int | None = None

    # Algorithm configuration
    use_kalman: bool = True
    use_spatial_index: bool = True
    bidirectional_matching: bool = True

    # Threshold configuration
    thresholds: dict = field(default_factory=lambda: {
        "appearance": 0.4764,
        "distance": 99.18,
        "size": 0.3143,
    })
    threshold_tolerance: float = 0.3
    get_gt_thresholds: bool = False
    gt_thresholds: str = "hill/train"

    # Weight configuration
    weight_appearance: float = 0.2
    weight_euclidean_distance: float = 0.2
    weight_kalman_distance: float = 0.5
    weight_size: float = 0.1

    # Track management
    max_age: int = 3
    min_iceberg_id_count: int = 1
    min_iceberg_size: float = 0.0

    # Kalman filter parameters
    process_noise: float = 10.0
    measurement_noise: float = 18.0

# ============================================================================
# TRACK REPRESENTATION
# ============================================================================

class IcebergTrack:
    """
    Represents an individual iceberg track with Kalman filter state estimation.

    Each track maintains:
    - Unique identifier (track_id)
    - Kalman filter for motion prediction (optional)
    - Match history and confidence metrics
    - Full detection history for output

    The Kalman filter models iceberg motion as constant velocity in 2D:
        State: [x, y, vx, vy, w, h]
        - (x, y): Center position
        - (vx, vy): Velocity in pixels/frame
        - (w, h): Bounding box dimensions

    This model assumes icebergs move smoothly with relatively constant velocity,
    which is reasonable for the timelapse timescales used (minutes to hours
    between frames).

    Attributes:
        track_id (int): Unique track identifier (never reused)
        config (IcebergTrackingConfig): Reference to tracking configuration
        kf (KalmanFilter | None): Kalman filter object (None if disabled)
        hits (int): Number of successful matches (confidence metric)
        age (int): Total frames since track creation
        time_since_update (int): Frames since last detection
        history (list): List of (frame_id, detection_id, bbox, confidence) tuples
        last_bbox (list): Most recent bounding box [x, y, w, h]
        last_detection (dict): Most recent detection dictionary
        predicted_bbox (list): Predicted position for current frame

    Methods:
        predict(): Advance Kalman filter, return predicted bbox
        update(detection, frame_id): Update with new detection
        get_state(): Get current estimated state
        get_velocity(): Get current velocity estimate
        get_uncertainty(): Get position uncertainty radius
    """

    def __init__(self, initial_detection, track_id, frame_id, config, distance_threshold):
        """
        Initialize a new track from an unmatched detection.

        Creates a track with identity track_id, initializes Kalman filter
        (if enabled), and stores the first detection in history.

        Args:
            initial_detection (dict): Detection dict with keys:
                - 'bbox': [x, y, w, h] bounding box
                - 'id': Detection ID within frame
                - 'conf': Detection confidence score
            track_id (int): Unique track identifier
            frame_id (str): Frame identifier (e.g., "000001")
            config (IcebergTrackingConfig): Tracking configuration
        """
        self.track_id = track_id
        self.config = config
        self.hits = 1  # Start with 1 hit (initial detection)
        self.age = 1  # Age in frames
        self.time_since_update = 0  # Frames since last match

        # Store complete detection history: (frame_id, det_id, bbox, confidence)
        self.history = [(frame_id, initial_detection['id'], initial_detection['bbox'], initial_detection['conf'])]
        self.last_bbox = initial_detection['bbox']
        self.last_detection = initial_detection
        self.distance_threshold = distance_threshold

        # Initialize Kalman filter if enabled
        if config.use_kalman:
            self.kf = self._init_kalman_filter(initial_detection['bbox'], config)
        else:
            self.kf = None
            # Without Kalman, predicted position is just last position
            self.predicted_bbox = initial_detection['bbox']

    def _init_kalman_filter(self, bbox, config):
        """
        Initialize Kalman filter for position and velocity estimation.

        Creates a 6-state Kalman filter with constant velocity motion model:

        Args:
            bbox (list): Initial bounding box [x, y, w, h]
            config (IcebergTrackingConfig): Configuration with noise parameters

        Returns:
            KalmanFilter: Initialized filter ready for prediction/update
        """
        # Create 6-state, 4-measurement Kalman filter
        kf = KalmanFilter(dim_x=6, dim_z=4)

        # State transition matrix F (constant velocity model)
        # x_k+1 = F * x_k
        kf.F = np.array([
            [1, 0, 1, 0, 0, 0],  # x_new = x + vx
            [0, 1, 0, 1, 0, 0],  # y_new = y + vy
            [0, 0, 1, 0, 0, 0],  # vx_new = vx (constant)
            [0, 0, 0, 1, 0, 0],  # vy_new = vy (constant)
            [0, 0, 0, 0, 1, 0],  # w_new = w (no size change)
            [0, 0, 0, 0, 0, 1]  # h_new = h (no size change)
        ])

        # Measurement function H (observe position and size)
        # z_k = H * x_k
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Measure x
            [0, 1, 0, 0, 0, 0],  # Measure y
            [0, 0, 0, 0, 1, 0],  # Measure w
            [0, 0, 0, 0, 0, 1]  # Measure h
        ])

        # Process noise covariance Q (motion uncertainty)
        q_pos = config.process_noise
        q_vel = config.process_noise  # Same as position noise (simplified)

        kf.Q = np.diag([
            q_pos,  # x position
            q_pos,  # y position
            q_vel,  # vx velocity
            q_vel,  # vy velocity
            q_pos,  # w size
            q_pos,  # h size
        ])

        # Measurement noise covariance R (detection uncertainty)
        r = config.measurement_noise

        kf.R = np.diag([
            r,  # x measurement
            r,  # y measurement
            r,  # w measurement
            r,  # h measurement
        ])

        # Initial state covariance P (initial uncertainty)
        # Start with high uncertainty since we don't know velocity
        kf.P = np.eye(6) * 100

        # Initial state vector [x, y, vx, vy, w, h]
        x, y, w, h = bbox
        kf.x = np.array([x, y, 0, 0, w, h])  # Initialize with zero velocity

        return kf

    def predict(self):
        """
        Predict the next state using Kalman filter or last position.

        Advances the Kalman filter by one time step to predict where the
        iceberg will be in the current frame based on its estimated velocity.

        If Kalman filtering is disabled, simply returns the last known position.

        Returns:
            list: Predicted bounding box [x, y, w, h]

        Kalman Prediction Steps:
            1. Predict state: x_pred = F * x
            2. Predict covariance: P_pred = F * P * F^T + Q
            3. Extract position and size from predicted state
        """
        if self.kf is not None:
            # Kalman prediction step
            self.kf.predict()
            state = self.kf.x
            # Extract predicted bbox from state [x, y, vx, vy, w, h]
            predicted_bbox = [state[0], state[1], state[4], state[5]]
        else:
            # No motion model - use last known position
            predicted_bbox = self.last_bbox

        # Store for use in matching
        self.predicted_bbox = predicted_bbox

        # Update age counters
        self.age += 1
        self.time_since_update += 1

        return predicted_bbox

    def update(self, detection, frame_id):
        """
        Update track with new detection using Kalman filter.

        When a track is successfully matched to a detection, this method
        updates the Kalman filter with the measurement and stores the
        detection in history.

        Args:
            detection (dict): Matched detection with keys:
                - 'bbox': [x, y, w, h]
                - 'id': Detection ID
                - 'conf': Confidence score
            frame_id (str): Current frame identifier
        """
        bbox = detection['bbox']

        if self.kf is not None:
            # Kalman update (correction) step
            # Measurement vector: [x, y, w, h]
            measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            self.kf.update(measurement)

        # Update track state
        self.last_bbox = bbox
        self.last_detection = detection
        self.hits += 1  # Increment successful match counter
        self.time_since_update = 0  # Reset missing counter

        # Store complete detection data in history
        self.history.append((frame_id, detection['id'], bbox, detection['conf']))

    def get_state(self):
        """
        Get current estimated state (position and size).

        Returns the current best estimate of the iceberg's position and size
        from the Kalman filter, or the last observed values if Kalman is disabled.

        Returns:
            list: Current estimated bbox [x, y, w, h]
        """
        if self.kf is not None:
            state = self.kf.x
            # Extract position and size from state vector
            return [state[0], state[1], state[4], state[5]]
        else:
            return self.last_bbox

    def get_velocity(self):
        """
        Get current velocity estimate from Kalman filter.

        Returns the estimated velocity in pixels per frame. This represents
        how fast and in what direction the iceberg is moving.

        Returns:
            tuple: (vx, vy) velocity in pixels/frame
                - vx: Horizontal velocity (positive = right)
                - vy: Vertical velocity (positive = down)
                - (0, 0) if Kalman filtering is disabled
        """
        if self.kf is not None:
            # Extract velocity from state vector [x, y, vx, vy, w, h]
            return (self.kf.x[2], self.kf.x[3])
        else:
            return (0, 0)

    def get_uncertainty(self):
        """
        Get position uncertainty radius from Kalman filter covariance.

        The uncertainty represents how confident we are about the track's
        position. Higher uncertainty suggests the filter is less certain,
        which can happen when:
        - Track is new (limited history)
        - Track was recently lost (prediction without updates)
        - Motion is unpredictable (high process noise)

        This uncertainty is used to adaptively adjust the search radius
        when looking for candidate detections.

        Returns:
            float: Uncertainty radius in pixels (2-sigma ellipse)
                - Small values (~20-50): High confidence
                - Large values (~100-200): Low confidence
                - Falls back to distance threshold if Kalman disabled
        """
        if self.kf is not None:
            # Extract position covariance from P matrix
            cov_xx = self.kf.P[0, 0]  # Variance in x
            cov_yy = self.kf.P[1, 1]  # Variance in y
            # Return 2-sigma radius (95% confidence ellipse)
            return 2 * np.sqrt(cov_xx + cov_yy)
        else:
            # No Kalman filter - use configured distance threshold
            return self.distance_threshold


# ============================================================================
# SPATIAL INDEXING
# ============================================================================

class SpatialIndex:
    """
    Grid-based spatial index for efficient nearest neighbor queries.

    This data structure dramatically speeds up candidate selection by
    organizing detections into a 2D grid. Instead of checking every detection,
    we only check detections in nearby grid cells.

    Grid Structure:
        - Image space divided into cells (e.g., 100x100 pixels)
        - Each detection assigned to cell based on its center
        - Query checks only cells within search radius

    Example:
        Scene with 2000 icebergs, search radius 150 pixels, cell size 100 pixels:
        - Naive: Check all 2000 icebergs = 2000 distance computations
        - Spatial index: Check ~20-30 icebergs in nearby cells = 25 distance computations
        - Speedup: 2000/25 = 80x faster!

    Attributes:
        cell_size (int): Size of grid cells in pixels
        index (dict): Mapping from (cell_x, cell_y) to list of (det_id, det_data)

    Methods:
        build(detections): Populate index from detections
        query_radius(position, radius): Get detections within radius
    """

    def __init__(self, cell_size=100):
        """
        Initialize spatial index with specified cell size.

        Args:
            cell_size (int): Size of grid cells in pixels (default: 100)
                - Smaller cells: More precise, but more cells to check
                - Larger cells: Less precise, but fewer cells to check
                - Rule of thumb: cell_size ≈ typical search radius
        """
        self.cell_size = cell_size
        self.index = defaultdict(list)  # Maps (cell_x, cell_y) -> [(det_id, det_data), ...]

    def build(self, detections):
        """
        Build spatial index from detections.

        Clears any existing index and populates it with the provided detections.
        Each detection is assigned to a grid cell based on its center position.

        Args:
            detections (dict): Detections keyed by det_id
                Format: {det_id: {'bbox': [x, y, w, h], ...}}
        """
        self.index.clear()

        for det_id, det_data in detections.items():
            x, y, w, h = det_data['bbox']
            # Use center point for indexing
            center_x = x + w / 2
            center_y = y + h / 2

            # Compute grid cell indices (integer division)
            cell_x = int(center_x // self.cell_size)
            cell_y = int(center_y // self.cell_size)

            # Add detection to cell
            self.index[(cell_x, cell_y)].append((det_id, det_data))

    def query_radius(self, position, radius):
        """
        Query all detections within radius of a position.

        Uses grid structure to efficiently find nearby detections:
        1. Determine which cells could contain results
        2. Collect detections from those cells
        3. Filter by actual distance

        Args:
            position (tuple): (x, y) center position to query
            radius (float): Search radius in pixels

        Returns:
            list: List of (det_id, det_data) tuples within radius
        """
        x, y = position

        # Compute how many cells to check in each direction
        r_cells = int(np.ceil(radius / self.cell_size))

        # Compute center cell
        center_cell_x = int(x // self.cell_size)
        center_cell_y = int(y // self.cell_size)

        # Collect candidates from nearby cells
        candidates = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                cell = (center_cell_x + dx, center_cell_y + dy)
                candidates.extend(self.index.get(cell, []))

        # Filter by actual Euclidean distance
        filtered = []
        for det_id, det_data in candidates:
            det_x, det_y, det_w, det_h = det_data['bbox']
            det_center_x = det_x + det_w / 2
            det_center_y = det_y + det_h / 2

            # Compute distance from query position to detection center
            dist = np.sqrt((det_center_x - x) ** 2 + (det_center_y - y) ** 2)

            if dist <= radius:
                filtered.append((det_id, det_data))

        return filtered


# ============================================================================
# MAIN TRACKING ORCHESTRATOR
# ============================================================================

class IcebergTracker:
    """
    Main orchestrator for iceberg multi-object tracking with Kalman filtering.

    This class coordinates the complete tracking pipeline across sequences and
    frames. It manages track creation, matching, updating, and deletion, and
    handles output generation.

    Responsibilities:
        - Sequence iteration and data loading
        - Frame-by-frame tracking loop
        - Track lifecycle management
        - Matching algorithm execution
        - Similarity computation with multi-feature fusion
        - Results output and post-processing

    Attributes:
        config (IcebergTrackingConfig): Complete configuration
        dataset (str): Dataset name/path
        sequences (dict): Sequence name to paths mapping
        tracks (list): Currently active tracks
        next_track_id (int): Next available track ID (monotonic)
        frame_count (int): Number of frames processed
        total_matches (int): Total successful matches
        total_detections (int): Total detections processed

    Methods:
        track(): Main entry point - processes all sequences
        _process_sequence(): Process single sequence frame-by-frame
        _track_frame(): Process single frame
        _compute_similarity(): Compute multi-feature similarity score
        _matching_bidirectional(): Mutual best match algorithm
        _matching_pure_greedy(): Global greedy matching algorithm
        _save_tracking_results(): Write results to file
    """

    def __init__(self, config: IcebergTrackingConfig):
        """
        Initialize tracker with configuration.

        Sets up all necessary state and loads sequence information.

        Args:
            config (IcebergTrackingConfig): Complete tracking configuration
        """
        self.config = config
        self.dataset = config.dataset
        self.sequences = get_sequences(self.dataset)
        self.thresholds = config.thresholds
        try:
            self.thresholds = get_gt_thresholds(config.gt_thresholds) if config.get_gt_thresholds else self.thresholds
        except:
            logger.info(f"\nGround truth dataset {self.config.gt_thresholds} not found. Using default thresholds instead.")

        # Track management state
        self.tracks = []  # Currently active tracks
        self.next_track_id = 1  # Monotonic track ID counter

        # Statistics
        self.frame_count = 0
        self.total_matches = 0
        self.total_detections = 0

        # Print configuration for transparency
        self._print_configuration()

    def _print_configuration(self):
        """
        Print comprehensive configuration summary.

        Displays all relevant parameters in an organized format for easy
        verification and reproducibility.
        """
        logger.info("\n📄 Iceberg Tracking Configuration")
        logger.info("=" * 60)
        logger.info(f"Dataset:                    {self.dataset}")
        logger.info(f"Sequences:                  {', '.join(self.sequences.keys())}")

        logger.info("\nAlgorithm Features:")
        logger.info(f"  Kalman filtering:         {self.config.use_kalman}")
        logger.info(f"  Spatial indexing:         {self.config.use_spatial_index}")
        logger.info(
            f"  Matching algorithm:       {'Bidirectional' if self.config.bidirectional_matching else 'Pure Greedy'}")

        logger.info("\nTrack Management:")
        logger.info(f"  Max age (frames):         {self.config.max_age}")
        logger.info(f"  Min track length:         {self.config.min_iceberg_id_count}")

        logger.info("\nSimilarity Thresholds:")
        logger.info(f"  Appearance:               {self.thresholds['appearance']:.4f}")
        logger.info(f"  Distance:                 {self.thresholds['distance']:.0f}px")
        logger.info(f"  Size:                     {self.thresholds['size']:.4f}")

        logger.info("\nFeature Weights:")
        logger.info(f"  Appearance:               {self.config.weight_appearance:.2f}")
        logger.info(f"  Euclidean Distance:       {self.config.weight_euclidean_distance:.2f}")
        logger.info(f"  Kalman Distance:          {self.config.weight_kalman_distance:.2f}")
        logger.info(f"  Size:                     {self.config.weight_size:.2f}")
        logger.info("=" * 60)

    def track(self):
        """
        Main tracking pipeline - processes all sequences.

        Iterates through all sequences in the dataset, loads required data
        (detections and embeddings), performs tracking, and saves results.
        """
        for sequence_name, paths in self.sequences.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing sequence: {sequence_name}")
            logger.info(f"{'=' * 60}")

            # Check required files exist
            if not paths["detections"].exists():
                logger.warning(f"⚠ Warning: No det.txt found, skipping...")
                continue

            if not paths["det_embeddings"].exists():
                logger.warning(f"⚠ Warning: No embeddings.pt found, skipping...")
                continue

            # Setup output directory
            base_path = str(paths["images"]).split("/images")[0]
            track_dir = os.path.join(base_path, "tracking")
            os.makedirs(track_dir, exist_ok=True)

            # Reset track state for new sequence
            self.tracks = []
            self.next_track_id = 1
            self.frame_count = 0

            # Load data
            icebergs_by_frame = load_icebergs_by_frame(paths["detections"])
            logger.info(f"\nLoading embeddings from {paths['det_embeddings']}...")
            iceberg_embeddings = torch.load(paths["det_embeddings"])
            logger.info(f"Loaded {len(iceberg_embeddings)} embeddings")

            # Process frames
            all_tracking_results = self._process_sequence(
                icebergs_by_frame,
                iceberg_embeddings
            )

            # Save results
            self._save_tracking_results(all_tracking_results, paths["tracking"])

        logger.info(f"\n{'=' * 60}")
        logger.info("Tracking complete!")
        logger.info(f"{'=' * 60}")

    def _process_sequence(self, icebergs_by_frame, embeddings):
        """
        Process entire sequence frame by frame with progress tracking.

        Args:
            icebergs_by_frame (dict): Detections organized by frame
            embeddings (dict): Pre-computed appearance embeddings

        Returns:
            list: All tracking results (list of dicts)
        """
        frames = sorted(icebergs_by_frame.keys())

        # Determine processing range
        if self.config.seq_length_limit is None:
            end_frame = len(frames)
        else:
            end_frame = min(self.config.seq_length_limit, len(frames))

        start_frame = self.config.seq_start_index
        all_results = []

        logger.info(f"\nProcessing {end_frame - start_frame} frames...")

        # Create progress bar
        progress_bar = tqdm(
            range(start_frame, end_frame),
            desc="Tracking frames",
            unit="frame"
        )

        # Process each frame
        for frame_idx in progress_bar:
            frame_id = frames[frame_idx]
            detections = icebergs_by_frame[frame_id]

            # Track one frame
            frame_results = self._track_frame(
                frame_id,
                detections,
                embeddings
            )

            all_results.extend(frame_results)

            # Update progress bar with live statistics
            progress_bar.set_postfix({
                'tracks': len(self.tracks),
                'matches': len(frame_results)
            })

        return all_results

    def _track_frame(self, frame_id, detections, embeddings):
        """
        Track icebergs in a single frame - the core tracking algorithm.

        This method implements the complete tracking pipeline for one frame:
        1. Predict: Use Kalman filters to predict new positions
        2. Index: Build spatial index for fast candidate lookup
        3. Match: Find best track-detection correspondences
        4. Update: Update matched tracks
        5. Delete: Remove old tracks
        6. Create: Initialize new tracks
        7. Output: Generate results

        Args:
            frame_id (str): Current frame identifier (e.g., "000001")
            detections (dict): Detections in this frame {det_id: det_data}
            embeddings (dict): Appearance embeddings {"{frame}_{id}": tensor}

        Returns:
            list: Tracking results for this frame (list of dicts)
                Each dict contains: frame_id, track_id, bbox, detection_id, confidence
        """
        self.frame_count += 1

        # Step 1: Predict new positions for all active tracks
        predictions = []
        for track in self.tracks:
            predicted_bbox = track.predict()  # Advance Kalman filter
            predictions.append((track, predicted_bbox))

        # Step 2: Build spatial index for fast candidate lookup (if enabled)
        if self.config.use_spatial_index:
            spatial_index = SpatialIndex(cell_size=100)
            spatial_index.build(detections)
        else:
            spatial_index = None

        # Step 3: Match tracks to detections using configured algorithm
        if self.config.bidirectional_matching:
            matches, unmatched_tracks, unmatched_dets = self._matching_bidirectional(
                frame_id, detections, embeddings, spatial_index
            )
        else:
            matches, unmatched_tracks, unmatched_dets = self._matching_pure_greedy(
                frame_id, detections, embeddings, spatial_index
            )

        # Step 4: Update matched tracks with new detections
        for track, detection in matches:
            track.update(detection, frame_id)

        # Step 5: Handle unmatched tracks (age them out if too old)
        tracks_to_remove = []
        for track in unmatched_tracks:
            if track.time_since_update > self.config.max_age:
                tracks_to_remove.append(track)

        # Remove dead tracks
        for track in tracks_to_remove:
            self.tracks.remove(track)

        # Step 6: Create new tracks for unmatched detections
        for detection in unmatched_dets:
            # Apply size filter to avoid tracking tiny noise
            bbox = detection['bbox']
            size = bbox[2] * bbox[3]
            if size >= self.config.min_iceberg_size:
                new_track = IcebergTrack(
                    detection,
                    self.next_track_id,
                    frame_id,
                    self.config,
                    distance_threshold=self.thresholds["distance"]
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Step 7: Generate tracking results for MATCHED tracks only
        frame_results = []
        for track in self.tracks:
            # Only output if track was updated this frame
            if track.time_since_update == 0:  # Was matched this frame
                frame_results.append({
                    'frame_id': frame_id,
                    'track_id': track.track_id,
                    'bbox': track.last_bbox,
                    'detection_id': track.last_detection['id'],
                    'confidence': track.last_detection['conf']
                })

        return frame_results


    def _compute_similarity(self, track, detection, features_a, features_b):
        """
        Compute multi-feature similarity score with Kalman-predicted distance.

        This is the core similarity computation that fuses multiple features:
        1. **Distance**: Spatial proximity (Kalman prediction + last position blend)
        2. **Appearance**: Visual similarity from deep learning embeddings
        3. **Size**: Bounding box area consistency

        Args:
            track (IcebergTrack): Track with Kalman prediction
            detection (dict): Detection with bbox and features
            features_a (torch.Tensor): Track appearance embedding
            features_b (torch.Tensor): Detection appearance embedding

        Returns:
            float | None: Weighted similarity score [0, 1] or None if filtered out
        """
        # Get last known position (conservative fallback)
        last_iceberg = {'bbox': track.last_bbox}
        # Current detection
        detection_iceberg = {'bbox': detection['bbox']}

        # Size similarity (medium cost)
        size_similarity = get_size_similarity(last_iceberg, detection_iceberg)

        if size_similarity >= self.thresholds['size'] * (1 - self.config.threshold_tolerance):
            # Appearance similarity (expensive - only compute if needed)
            appearance_similarity = get_appearance_similarity(features_a, features_b, "cpu")

            if appearance_similarity >= self.thresholds['appearance'] * (1 - self.config.threshold_tolerance):
                # Compute distance to last known position
                distance_eucl = get_distance(last_iceberg, detection_iceberg)
                # Get Kalman predicted position (motion-aware)
                predicted_bbox = track.predicted_bbox
                predicted_iceberg = {'bbox': predicted_bbox}
                # Compute distance to Kalman predicted position
                distance_kalman = get_distance(predicted_iceberg, detection_iceberg)

                # Normalize distance to [0, 1] (1 = close, 0 = far)
                kalman_distance_norm = 1 - min_max_normalize(
                    distance_kalman, 0, self.thresholds['distance']
                )

                eucl_distance_norm = 1 - min_max_normalize(
                    distance_eucl, 0, self.thresholds['distance']
                )

                size_similarity = min_max_normalize(
                    size_similarity, self.thresholds['size'], 1.0
                )

                appearance_similarity = min_max_normalize(
                    appearance_similarity, self.thresholds['appearance'], 1.0
                )

                # Compute weighted score with configured weights
                weighted_score = get_score(
                    appearance_similarity,
                    eucl_distance_norm,
                    kalman_distance_norm,
                    size_similarity,
                    self.config.weight_appearance,
                    self.config.weight_euclidean_distance,
                    self.config.weight_kalman_distance,
                    self.config.weight_size
                )
                return weighted_score

        # Filtered out at some threshold
        return None

    def _matching_bidirectional(self, frame_id, detections, embeddings, spatial_index=None):
        """
        Bidirectional matching algorithm (mutual best match required).

        This algorithm requires both track and detection to prefer each other
        for a match to occur. It's more conservative than pure greedy matching,
        resulting in fewer false matches but potentially more fragmented tracks.

        Args:
            frame_id (str): Current frame identifier
            detections (dict): Detections {det_id: det_data}
            embeddings (dict): Appearance embeddings
            spatial_index (SpatialIndex | None): Optional spatial index

        Returns:
            tuple: (matches, unmatched_tracks, unmatched_dets)
                - matches: List of (track, detection) tuples
                - unmatched_tracks: List of tracks without matches
                - unmatched_dets: List of detections without matches
        """
        # Tracking structures for best matches
        track_best_match = {}  # track -> (detection, similarity)
        det_best_match = {}  # det_id -> (track, similarity)

        # Convert detections to list for easier iteration
        all_detections = list(detections.values())

        # Phase 1: Find best match for each track
        for track in self.tracks:
            # Get predicted position for spatial query
            pred_bbox = track.predicted_bbox
            pred_x, pred_y, pred_w, pred_h = pred_bbox
            pred_center = (pred_x + pred_w / 2, pred_y + pred_h / 2)

            # Determine search radius (adaptive if using Kalman)
            if self.config.use_kalman:
                search_radius = track.get_uncertainty()
                search_radius = max(search_radius, self.thresholds['distance'] * (1 + self.config.threshold_tolerance))
            else:
                search_radius = self.thresholds['distance'] * (1 + self.config.threshold_tolerance)

            # Get candidate detections within search radius
            if spatial_index is not None:
                candidates = spatial_index.query_radius(pred_center, search_radius)
            else:
                # No spatial index - check all detections (slow)
                candidates = [(d['id'], d) for d in all_detections]

            # Find best match among candidates
            best_det = None
            best_similarity = 0

            for det_id, detection in candidates:
                # Get embeddings for similarity computation
                track_emb_key = f"{track.history[-1][0]}_{track.history[-1][1]}"
                det_emb_key = f"{frame_id}_{det_id}"

                track_embedding = embeddings.get(track_emb_key)
                det_embedding = embeddings.get(det_emb_key)

                if track_embedding is None or det_embedding is None:
                    continue

                # Compute similarity (uses Kalman predicted position)
                similarity = self._compute_similarity(
                    track,
                    detection,
                    track_embedding,
                    det_embedding
                )

                if similarity is not None and similarity > best_similarity:
                    best_similarity = similarity
                    best_det = detection

            # Store best match for this track
            if best_det is not None:
                track_best_match[track] = (best_det, best_similarity)

                # Also track best match FROM detection's perspective
                det_id = best_det['id']
                if det_id not in det_best_match or best_similarity > det_best_match[det_id][1]:
                    det_best_match[det_id] = (track, best_similarity)

        # Phase 2: Bidirectional matching (mutual preference)
        matches = []
        matched_tracks = set()
        matched_det_ids = set()

        for track, (best_det, similarity) in track_best_match.items():
            det_id = best_det['id']

            # Check if detection also prefers this track
            if det_id in det_best_match:
                det_best_track, _ = det_best_match[det_id]

                if det_best_track == track:
                    # Bidirectional match! Both prefer each other
                    matches.append((track, best_det))
                    matched_tracks.add(track)
                    matched_det_ids.add(det_id)

        # Find unmatched entities
        unmatched_tracks = [t for t in self.tracks if t not in matched_tracks]
        unmatched_dets = [d for d in all_detections if d['id'] not in matched_det_ids]

        return matches, unmatched_tracks, unmatched_dets

    def _matching_pure_greedy(self, frame_id, detections, embeddings, spatial_index=None):
        """
        Pure greedy matching based on global similarity ranking.

        This algorithm collects ALL candidate (track, detection) pairs with
        their similarities, sorts them globally by similarity (highest first),
        and greedily assigns matches without requiring bidirectional agreement.

        Args:
            frame_id (str): Current frame identifier
            detections (dict): Detections {det_id: det_data}
            embeddings (dict): Appearance embeddings
            spatial_index (SpatialIndex | None): Optional spatial index

        Returns:
            tuple: (matches, unmatched_tracks, unmatched_dets)
                - matches: List of (track, detection) tuples
                - unmatched_tracks: List of tracks without matches
                - unmatched_dets: List of detections without matches
        """
        # Convert detections to list
        all_detections = list(detections.values())

        # Collect ALL candidate matches with similarities
        all_candidates = []  # List of (similarity, track, detection)

        # Phase 1: Compute all candidate similarities
        for track in self.tracks:
            # Get predicted position for spatial query
            pred_bbox = track.predicted_bbox
            pred_x, pred_y, pred_w, pred_h = pred_bbox
            pred_center = (pred_x + pred_w / 2, pred_y + pred_h / 2)

            # Determine search radius (adaptive if using Kalman)
            if self.config.use_kalman:
                search_radius = track.get_uncertainty()
                search_radius = max(search_radius, self.thresholds['distance'] * (1 + self.config.threshold_tolerance))
            else:
                search_radius = self.thresholds['distance'] * (1 + self.config.threshold_tolerance)

            # Get candidate detections
            if spatial_index is not None:
                candidates = spatial_index.query_radius(pred_center, search_radius)
            else:
                # No spatial index - check all detections
                candidates = [(d['id'], d) for d in all_detections]

            # Compute similarity for each candidate
            for det_id, detection in candidates:
                # Get embeddings
                track_emb_key = f"{track.history[-1][0]}_{track.history[-1][1]}"
                det_emb_key = f"{frame_id}_{det_id}"

                track_embedding = embeddings.get(track_emb_key)
                det_embedding = embeddings.get(det_emb_key)

                if track_embedding is None or det_embedding is None:
                    continue

                # Compute similarity (uses Kalman predicted position)
                similarity = self._compute_similarity(
                    track,
                    detection,
                    track_embedding,
                    det_embedding
                )

                if similarity is not None:
                    # Store this candidate match
                    all_candidates.append((similarity, track, detection))

        # Phase 2: KEY STEP - Sort ALL candidates by similarity (HIGHEST first)
        # This is what makes pure greedy work better than bidirectional in dense scenes
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Phase 3: Greedy assignment - process best matches first
        matches = []
        matched_tracks = set()
        matched_det_ids = set()

        for similarity, track, detection in all_candidates:
            det_id = detection['id']

            # Skip if already matched (one-to-one constraint)
            if track in matched_tracks or det_id in matched_det_ids:
                continue

            # Assign this match (no bidirectional check!)
            matches.append((track, detection))
            matched_tracks.add(track)
            matched_det_ids.add(det_id)

        # Find unmatched entities
        unmatched_tracks = [t for t in self.tracks if t not in matched_tracks]
        unmatched_dets = [d for d in all_detections if d['id'] not in matched_det_ids]

        return matches, unmatched_tracks, unmatched_dets

    def _save_tracking_results(self, all_results, output_path):
        """
        Save tracking results to file in MOTChallenge format.

        Applies post-processing filters:
        1. Minimum track length filter (min_iceberg_id_count)
        2. Removes tracks that are too short (likely false positives)

        MOTChallenge Format:
            <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<x>,<x>

        Args:
            all_results (list): List of result dicts from all frames
            output_path (Path): Path to save tracking file
        """
        logger.info(f"\nSaving tracking results...")

        # Compute track lengths for filtering
        track_lengths = defaultdict(int)
        for result in all_results:
            track_lengths[result['track_id']] += 1

        # Identify valid tracks (meet minimum length requirement)
        valid_track_ids = {
            tid for tid, length in track_lengths.items()
            if length >= self.config.min_iceberg_id_count
        }

        # Write to file in MOTChallenge format
        with open(output_path, 'w') as f:
            for result in all_results:
                if result['track_id'] not in valid_track_ids:
                    continue

                x, y, w, h = result['bbox']
                f.write(
                    f"{result['frame_id']},{result['track_id']},"
                    f"{x},{y},{w},{h},"
                    f"{result['confidence']},1,-1,-1\n"
                )

        # Sort by frame number for easier analysis
        sort_file(output_path)

        # Log statistics
        filtered_out = len(all_results) - sum(1 for r in all_results if r['track_id'] in valid_track_ids)
        logger.info(f"Total tracking entries: {len(all_results)}")
        logger.info(f"Filtered out (too short): {filtered_out}")
        logger.info(f"Valid tracks: {len(valid_track_ids)}")
        logger.info(f"Saved to: {output_path}")