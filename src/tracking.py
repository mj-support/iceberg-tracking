import numpy as np
import os
import torch
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

from utils.feature_extraction import get_distance, get_size_similarity, get_appearance_similarity, \
    min_max_normalize, get_score
from utils.helpers import load_icebergs_by_frame, sort_file, get_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Iceberg Multi-Object Tracking Pipeline

This module implements an advanced multi-object tracking system for icebergs across
time-lapse imagery sequences. The system combines computer vision and machine learning 
techniques to maintain consistent iceberg identities over time.

Key Features:
    - Kalman filtering for motion prediction and state estimation
    - Spatial indexing for efficient candidate selection (50-100x speedup)
    - Bidirectional and pure greedy matching algorithms
    - Vision Transformer-based appearance embeddings
    - Robust track management with age and confidence tracking
    - Handles occlusions and missed detections gracefully

Architecture:
    - IcebergTrackingConfig: Configuration dataclass for all tracking parameters
    - IcebergTrack: Individual track representation with Kalman filter
    - SpatialIndex: Grid-based spatial indexing for fast neighbor queries
    - IcebergTracker: Main tracker coordinating the full pipeline
"""


@dataclass
class IcebergTrackingConfig:
    """
    Configuration class for iceberg tracking with motion prediction.

    This dataclass encapsulates all parameters needed to configure the tracking
    pipeline, including data paths, algorithm choices, thresholds, weights, and
    Kalman filter parameters.

    Attributes:
        dataset (str): Name/path of the dataset to process
        seq_start_index (int): Starting frame index for processing
        seq_length_limit (int | None): Maximum number of frames to process (None = all)

        use_kalman (bool): Enable Kalman filtering for motion prediction
        use_spatial_index (bool): Enable spatial indexing for speedup
        bidirectional_matching (bool): Use bidirectional matching (True) or pure greedy (False)

        thresholds (dict): Dictionary of threshold values:
            - appearance: Minimum appearance similarity [0, 1]
            - distance: Maximum distance in pixels (search radius)
            - size: Minimum size similarity [0, 1]
            - match_score: Minimum overall weighted score [0, 1]

        weight_appearance (float): Weight for appearance similarity in [0, 1]
        weight_size (float): Weight for size similarity in [0, 1]
        weight_distance (float): Weight for distance similarity in [0, 1]

        max_age (int): Maximum frames without detection before track deletion
        min_hits (int): Minimum hits before track is considered confirmed
        min_iceberg_id_count (int): Minimum detections to keep a track in results
        min_iceberg_size (float): Minimum iceberg bounding box area (pixels²)

        process_noise (float): Process noise for Kalman filter (motion uncertainty in pixels)
        measurement_noise (float): Measurement noise for Kalman filter (detection uncertainty in pixels)

        device (str): PyTorch device for computation ('cuda' or 'cpu')
    """
    # Data configuration
    dataset: str
    seq_start_index: int = 0
    seq_length_limit: int | None = None

    # Tracking algorithm configuration
    use_kalman: bool = True  # Enable motion prediction
    use_spatial_index: bool = True  # Enable spatial indexing
    bidirectional_matching: bool = True  # True: bidirectional matching, False: pure greedy

    # Threshold configuration
    thresholds: dict = field(default_factory=lambda: {
        "appearance": 0.3796,  # Minimum appearance similarity
        "distance": 122,  # Maximum distance in pixels (search radius)
        "size": 0.2933,  # Minimum size similarity
        "match_score": 0.4705  # Minimum overall match score
    })

    # Weight configuration (should sum to 1.0)
    weight_appearance: float = 0.35
    weight_size: float = 0.05
    weight_distance: float = 0.60

    # Track management
    max_age: int = 3  # Max frames without detection before track deletion
    min_hits: int = 2  # Min hits before track is confirmed
    min_iceberg_id_count: int = 2  # Minimum detections to keep a track
    min_iceberg_size: float = 0.0  # Minimum iceberg size

    # Kalman filter parameters
    process_noise: float = 15.0  # Motion model uncertainty (pixels)
    measurement_noise: float = 15.0  # Detection uncertainty (pixels)

    # Device configuration
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IcebergTrack:
    """
    Represents a single iceberg track with Kalman filtering for state estimation.

    An IcebergTrack maintains the history and state of a single iceberg across
    multiple frames. It uses a Kalman filter to predict future positions based
    on observed motion patterns and updates its state when matched with new
    detections.

    State Representation:
        - Position: (x, y) center coordinates in pixels
        - Velocity: (vx, vy) velocity in pixels/frame
        - Size: (w, h) bounding box width and height

    Attributes:
        track_id (int): Unique identifier for this track
        config (IcebergTrackingConfig): Tracking configuration
        kf (KalmanFilter | None): Kalman filter for state estimation (None if disabled)
        hits (int): Number of times this track has been matched with detections
        age (int): Total number of frames since track was created
        time_since_update (int): Number of frames since last detection match
        history (list): List of (frame_id, detection_id, bbox, confidence) tuples
        last_bbox (list): Most recent bounding box [x, y, w, h]
        last_detection (dict): Most recent detection data
        predicted_bbox (list): Predicted bounding box for current frame [x, y, w, h]

    Methods:
        predict(): Predict next state using Kalman filter
        update(): Update track with new detection
        get_state(): Get current estimated state
        get_velocity(): Get current velocity estimate
        get_uncertainty(): Get position uncertainty for adaptive search radius
    """

    def __init__(self, initial_detection, track_id, frame_id, config):
        """
        Initialize a new track with Kalman filter.

        Creates a new track starting from an initial detection. If Kalman filtering
        is enabled, initializes a Kalman filter with the detection's position and
        zero initial velocity.

        Args:
            initial_detection (dict): Detection dict containing:
                - 'bbox': [x, y, w, h] bounding box
                - 'id': detection ID
                - 'conf': confidence score
            track_id (int): Unique track ID assigned by tracker
            frame_id (int): Frame ID where track was initialized
            config (IcebergTrackingConfig): Tracking configuration
        """
        self.track_id = track_id
        self.config = config
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Store full detection data (including confidence)
        self.history = [(frame_id, initial_detection['id'], initial_detection['bbox'], initial_detection['conf'])]
        self.last_bbox = initial_detection['bbox']
        self.last_detection = initial_detection

        # Initialize Kalman filter if enabled
        if config.use_kalman:
            self.kf = self._init_kalman_filter(initial_detection['bbox'], config)
        else:
            self.kf = None
            self.predicted_bbox = initial_detection['bbox']

    def _init_kalman_filter(self, bbox, config):
        """
        Initialize Kalman filter for position and velocity tracking.

        Creates a 6-state Kalman filter using a constant velocity motion model.
        The filter estimates position, velocity, and size of the iceberg.

        State Vector (6D):
            [x, y, vx, vy, w, h]
            - x, y: Center position (pixels)
            - vx, vy: Velocity (pixels/frame)
            - w, h: Bounding box width and height (pixels)

        Measurement Vector (4D):
            [x, y, w, h]
            - Direct observation of position and size

        Motion Model:
            - Constant velocity: position += velocity each frame
            - Size assumed constant

        Args:
            bbox (list): Initial bounding box [x, y, w, h]
            config (IcebergTrackingConfig): Configuration with noise parameters

        Returns:
            KalmanFilter: Initialized Kalman filter ready for prediction/update
        """
        kf = KalmanFilter(dim_x=6, dim_z=4)

        # State transition matrix (constant velocity model)
        # Describes how state evolves: x_new = F * x_old
        kf.F = np.array([
            [1, 0, 1, 0, 0, 0],  # x_new = x + vx
            [0, 1, 0, 1, 0, 0],  # y_new = y + vy
            [0, 0, 1, 0, 0, 0],  # vx_new = vx (constant)
            [0, 0, 0, 1, 0, 0],  # vy_new = vy (constant)
            [0, 0, 0, 0, 1, 0],  # w_new = w (constant)
            [0, 0, 0, 0, 0, 1]  # h_new = h (constant)
        ])

        # Measurement function (we observe x, y, w, h)
        # Describes how measurements relate to state: z = H * x
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # measure x
            [0, 1, 0, 0, 0, 0],  # measure y
            [0, 0, 0, 0, 1, 0],  # measure w
            [0, 0, 0, 0, 0, 1]  # measure h
        ])

        # Process noise covariance (uncertainty in motion model)
        # Higher values = less trust in motion model, more responsive to measurements
        q = config.process_noise
        kf.Q = np.eye(6) * q
        kf.Q[2:4, 2:4] *= 2  # Higher uncertainty in velocity (harder to estimate)

        # Measurement noise covariance (uncertainty in detections)
        # Higher values = less trust in measurements, smoother tracking
        r = config.measurement_noise
        kf.R = np.eye(4) * r

        # Initial state covariance (initial uncertainty)
        # High initial values allow filter to quickly adapt
        kf.P = np.eye(6) * 100

        # Initial state [x, y, vx, vy, w, h]
        # Start with zero velocity assumption
        x, y, w, h = bbox
        kf.x = np.array([x, y, 0, 0, w, h])

        return kf

    def predict(self):
        """
        Predict the next state using Kalman filter.

        Performs the prediction step of the Kalman filter, estimating where the
        iceberg will be in the current frame based on its previous state and
        motion model. Also increments age counters.

        If Kalman filtering is disabled, returns the last known position.

        Returns:
            list: Predicted bounding box [x, y, w, h]
        """
        if self.kf is not None:
            # Kalman prediction: x_pred = F * x
            self.kf.predict()
            state = self.kf.x
            predicted_bbox = [state[0], state[1], state[4], state[5]]
        else:
            # No motion model - use last position
            predicted_bbox = self.last_bbox

        self.predicted_bbox = predicted_bbox
        self.age += 1
        self.time_since_update += 1
        return predicted_bbox

    def update(self, detection, frame_id):
        """
        Update track with new detection.

        Performs the update/correction step of the Kalman filter, incorporating
        a new detection measurement to refine the state estimate. Also updates
        tracking statistics and history.

        Args:
            detection (dict): Detection dict containing:
                - 'bbox': [x, y, w, h] measured bounding box
                - 'id': detection ID
                - 'conf': confidence score
            frame_id (int): Current frame ID
        """
        bbox = detection['bbox']

        if self.kf is not None:
            # Kalman update: corrects prediction with measurement
            measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            self.kf.update(measurement)

        # Update track state
        self.last_bbox = bbox
        self.last_detection = detection
        self.hits += 1
        self.time_since_update = 0

        # Store full detection data including confidence
        self.history.append((frame_id, detection['id'], bbox, detection['conf']))

    def get_state(self):
        """
        Get current estimated state (bounding box).

        Returns the current state estimate from the Kalman filter, or the last
        known position if Kalman filtering is disabled.

        Returns:
            list: Current estimated bounding box [x, y, w, h]
        """
        if self.kf is not None:
            state = self.kf.x
            return [state[0], state[1], state[4], state[5]]
        else:
            return self.last_bbox

    def get_velocity(self):
        """
        Get current velocity estimate from Kalman filter.

        Returns the estimated velocity of the iceberg in pixels per frame.
        This can be used for trajectory analysis or motion-based filtering.

        Returns:
            tuple: (vx, vy) velocity in pixels/frame, or (0, 0) if Kalman disabled
        """
        if self.kf is not None:
            return (self.kf.x[2], self.kf.x[3])
        else:
            return (0, 0)

    def get_uncertainty(self):
        """
        Get position uncertainty (covariance) from Kalman filter.

        Computes a scalar uncertainty measure from the position covariance matrix.
        This is used to adaptively adjust the search radius: tracks with high
        uncertainty get larger search radii to account for prediction uncertainty.

        The uncertainty is computed as 2-sigma (95% confidence) radius of the
        position uncertainty ellipse.

        Returns:
            float: Uncertainty radius in pixels (2-sigma), or default distance 
                   threshold if Kalman is disabled
        """
        if self.kf is not None:
            # Extract position covariance (variance in x and y)
            cov_xx = self.kf.P[0, 0]
            cov_yy = self.kf.P[1, 1]
            # Return radius of uncertainty ellipse (2 sigma ~ 95% confidence)
            return 2 * np.sqrt(cov_xx + cov_yy)
        else:
            return self.config.thresholds['distance']


class SpatialIndex:
    """
    Spatial indexing for fast neighbor queries using grid-based hashing.

    Implements a 2D spatial hash grid to speed up candidate selection.
    The space is divided into uniform grid cells,
    and detections are indexed by their cell coordinates.

    Attributes:
        cell_size (float): Size of each grid cell in pixels
        index (defaultdict): Hash map from (cell_x, cell_y) to list of (det_id, det_data)

    Methods:
        build(): Build spatial index from detection data
        query_radius(): Query detections within radius of a point
    """

    def __init__(self, cell_size=100):
        """
        Initialize spatial index with specified cell size.

        The cell_size parameter controls the granularity of the grid. Smaller
        cells provide more precise queries but increase memory usage. Typically,
        cell_size should be comparable to the search radius.

        Args:
            cell_size (float): Size of grid cells in pixels (default: 100)
        """
        self.cell_size = cell_size
        self.index = defaultdict(list)

    def build(self, detections):
        """
        Build spatial index from detections.

        Clears any existing index and rebuilds it from the provided detections.
        Each detection is inserted into the grid cell corresponding to its
        center position.

        Args:
            detections (dict): Dictionary of {det_id: detection_data} where
                detection_data contains 'bbox': [x, y, w, h]

        Time Complexity: O(n) where n is number of detections
        """
        # Clear previous index
        self.index.clear()

        # Insert each detection into appropriate grid cell
        for det_id, det_data in detections.items():
            x, y, w, h = det_data['bbox']

            # Use center point for indexing
            center_x = x + w / 2
            center_y = y + h / 2

            # Compute grid cell coordinates
            cell_x = int(center_x // self.cell_size)
            cell_y = int(center_y // self.cell_size)

            # Insert into spatial hash
            self.index[(cell_x, cell_y)].append((det_id, det_data))

    def query_radius(self, position, radius):
        """
        Query detections within radius of position.

        Efficiently finds all detections whose center lies within the specified
        radius of the query position. Uses the grid structure to only check
        cells that could contain candidates, then filters by exact distance.

        Args:
            position (tuple): Query position (x, y) in pixels
            radius (float): Search radius in pixels

        Returns:
            list: List of (det_id, det_data) tuples for detections within radius
        """
        x, y = position

        # Determine how many cells to check in each direction
        r_cells = int(np.ceil(radius / self.cell_size))

        # Compute center cell coordinates
        center_cell_x = int(x // self.cell_size)
        center_cell_y = int(y // self.cell_size)

        # Collect candidates from nearby cells (square region)
        candidates = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                cell = (center_cell_x + dx, center_cell_y + dy)
                candidates.extend(self.index.get(cell, []))

        # Filter by actual Euclidean distance (refine square to circle)
        filtered = []
        for det_id, det_data in candidates:
            det_x, det_y, det_w, det_h = det_data['bbox']
            det_center_x = det_x + det_w / 2
            det_center_y = det_y + det_h / 2

            # Compute Euclidean distance
            dist = np.sqrt((det_center_x - x) ** 2 + (det_center_y - y) ** 2)
            if dist <= radius:
                filtered.append((det_id, det_data))

        return filtered


class IcebergTracker:
    """
    Main iceberg tracker coordinating the full multi-object tracking pipeline.

    This class orchestrates the entire tracking process across multiple sequences
    and frames. It manages track lifecycles, coordinates matching between tracks
    and detections, and outputs tracking results in MOT format.

    Pipeline Overview:
        1. Load detections and embeddings for each sequence
        2. For each frame:
           a. Predict new positions for all tracks (Kalman filter)
           b. Build spatial index from detections
           c. Match tracks to detections (bidirectional or greedy)
           d. Update matched tracks
           e. Delete old unmatched tracks
           f. Create new tracks for unmatched detections
           g. Output results for confirmed tracks
        3. Save tracking results in MOT format

    Attributes:
        config (IcebergTrackingConfig): Tracking configuration
        dataset (str): Dataset name/path
        device (torch.device): PyTorch device for computation
        sequences (dict): Dictionary of sequence paths
        tracks (list): List of active IcebergTrack objects
        next_track_id (int): Next available track ID
        frame_count (int): Number of frames processed in current sequence
        total_matches (int): Total number of matches (statistics)
        total_detections (int): Total number of detections (statistics)

    Methods:
        track(): Main entry point - process all sequences
        _process_sequence(): Process a single sequence
        _track_frame(): Track icebergs in a single frame
        _compute_similarity(): Compute weighted similarity between track and detection
        _matching_bidirectional(): Bidirectional matching algorithm
        _matching_pure_greedy(): Pure greedy matching algorithm
        _save_tracking_results(): Save results to file in MOT format
    """

    def __init__(self, config: IcebergTrackingConfig):
        """
        Initialize tracker with configuration.

        Sets up the tracker by loading sequence information and initializing
        tracking state. Prints a configuration summary for reference.

        Args:
            config (IcebergTrackingConfig): Complete tracking configuration
        """
        self.config = config
        self.dataset = config.dataset
        self.device = config.device
        self.sequences = get_sequences(self.dataset)

        # Track management
        self.tracks = []  # Active tracks
        self.next_track_id = 1

        # Statistics
        self.frame_count = 0
        self.total_matches = 0
        self.total_detections = 0

        self._print_configuration()

    def _print_configuration(self):
        """
        Print configuration summary to console.

        Displays all important configuration parameters in a formatted manner
        for easy verification and documentation.
        """
        logger.info("\n📄 Improved Iceberg Tracking Configuration")
        logger.info("=" * 60)
        logger.info(f"Dataset:                    {self.dataset}")
        logger.info(f"Sequences:                  {', '.join(self.sequences.keys())}")
        logger.info(f"Device:                     {self.device}")
        logger.info("\nAlgorithm Features:")
        logger.info(f"  Kalman filtering:         {self.config.use_kalman}")
        logger.info(f"  Spatial indexing:         {self.config.use_spatial_index}")
        logger.info(
            f"  Matching algorithm:       {'Bidirectional' if self.config.bidirectional_matching else 'Pure Greedy'}")
        logger.info("\nTrack Management:")
        logger.info(f"  Max age (frames):         {self.config.max_age}")
        logger.info(f"  Min hits:                 {self.config.min_hits}")
        logger.info(f"  Min track length:         {self.config.min_iceberg_id_count}")
        logger.info("\nSimilarity Thresholds:")
        logger.info(f"  Appearance:               {self.config.thresholds['appearance']:.4f}")
        logger.info(f"  Distance:                 {self.config.thresholds['distance']:.0f}px")
        logger.info(f"  Size:                     {self.config.thresholds['size']:.4f}")
        logger.info(f"  Match score:              {self.config.thresholds['match_score']:.4f}")
        logger.info("\nWeights:")
        logger.info(f"  Distance:   {self.config.weight_distance:.2f}")
        logger.info(f"  Appearance: {self.config.weight_appearance:.2f}")
        logger.info(f"  Size:       {self.config.weight_size:.2f}")
        logger.info("=" * 60)

    def track(self):
        """
        Main tracking pipeline - process all sequences.

        Entry point for the tracking system. Iterates through all sequences in
        the dataset, loads their data, performs tracking, and saves results.
        Each sequence is processed independently with its own track management.
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
        Process entire sequence frame by frame.

        Iterates through all frames in the sequence, performing tracking on each.
        Shows progress with tqdm progress bar.

        Args:
            icebergs_by_frame (dict): Dictionary mapping frame_id to detections
            embeddings (dict): Pre-computed appearance embeddings for all detections

        Returns:
            list: List of tracking result dictionaries, one per detection in a track
        """
        frames = sorted(icebergs_by_frame.keys())

        # Determine processing range based on configuration
        if self.config.seq_length_limit is None:
            end_frame = len(frames)
        else:
            end_frame = min(self.config.seq_length_limit, len(frames))

        start_frame = self.config.seq_start_index
        all_results = []

        logger.info(f"\nProcessing {end_frame - start_frame} frames...")

        # Create progress bar for visual feedback
        progress_bar = tqdm(
            range(start_frame, end_frame),
            desc="Tracking frames",
            unit="frame"
        )

        # Process each frame sequentially
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

            # Update progress bar with current statistics
            progress_bar.set_postfix({
                'tracks': len(self.tracks),
                'matches': len(frame_results)
            })

        return all_results

    def _track_frame(self, frame_id, detections, embeddings):
        """
        Track icebergs in a single frame.

        Implements the core tracking loop for one frame:
        1. Predict new positions (Kalman)
        2. Build spatial index
        3. Match tracks to detections
        4. Update matched tracks
        5. Delete old tracks
        6. Create new tracks
        7. Generate output for confirmed tracks

        Args:
            frame_id (int): Current frame ID
            detections (dict): Dictionary of detections {det_id: detection_data}
            embeddings (dict): Pre-computed appearance embeddings

        Returns:
            list: List of tracking results for this frame, each containing:
                - frame_id: Frame number
                - track_id: Track identifier
                - bbox: Bounding box [x, y, w, h]
                - detection_id: Original detection ID
                - confidence: Detection confidence score
        """
        self.frame_count += 1

        # Step 1: Predict new positions for all active tracks
        # Uses Kalman filter to estimate where each iceberg should be
        predictions = []
        for track in self.tracks:
            predicted_bbox = track.predict()
            predictions.append((track, predicted_bbox))

        # Step 2: Build spatial index for fast candidate queries
        # Dramatically speeds up matching in dense scenes
        if self.config.use_spatial_index:
            spatial_index = SpatialIndex(cell_size=100)
            spatial_index.build(detections)
        else:
            spatial_index = None

        # Step 3: Match tracks to detections
        # Choose algorithm based on configuration
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

        # Step 5: Handle unmatched tracks - delete if too old
        tracks_to_remove = []
        for track in unmatched_tracks:
            if track.time_since_update > self.config.max_age:
                tracks_to_remove.append(track)

        # Remove dead tracks from active list
        for track in tracks_to_remove:
            self.tracks.remove(track)

        # Step 6: Create new tracks for unmatched detections
        for detection in unmatched_dets:
            # Apply minimum size filter
            bbox = detection['bbox']
            size = bbox[2] * bbox[3]
            if size >= self.config.min_iceberg_size:
                new_track = IcebergTrack(
                    detection,
                    self.next_track_id,
                    frame_id,
                    self.config
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Step 7: Generate tracking results for confirmed tracks only
        frame_results = []
        for track in self.tracks:
            # Output retroactive history when track becomes confirmed
            if track.hits == self.config.min_hits and track.time_since_update == 0:
                # Track just got confirmed! Output entire history retroactively
                for hist_frame_id, hist_det_id, hist_bbox, hist_conf in track.history:
                    frame_results.append({
                        'frame_id': hist_frame_id,
                        'track_id': track.track_id,
                        'bbox': hist_bbox,
                        'detection_id': hist_det_id,
                        'confidence': hist_conf
                    })
            elif track.hits > self.config.min_hits and track.time_since_update == 0:
                # Already confirmed - just output current frame
                frame_results.append({
                    'frame_id': frame_id,
                    'track_id': track.track_id,
                    'bbox': track.last_bbox,
                    'detection_id': track.last_detection['id'],
                    'confidence': track.last_detection['conf']
                })

        return frame_results

    def _compute_similarity(self, iceberg_a, iceberg_b, features_a, features_b):
        """
        Compute weighted similarity score between two icebergs.

        Combines multiple similarity criteria (appearance, distance, size) into
        a single weighted score. Applies thresholds to filter out unlikely matches
        early for efficiency.

        The similarity computation follows this hierarchical approach:
        1. Compute raw features (distance, size similarity, appearance similarity)
        2. Normalize each feature to [0, 1] range
        3. Compute weighted combination
        4. Apply thresholds to filter out poor matches

        Thresholds are relaxed by factor of 1.5 to allow some flexibility.

        Args:
            iceberg_a (dict): First iceberg with 'bbox' key [x, y, w, h]
            iceberg_b (dict): Second iceberg with 'bbox' key [x, y, w, h]
            features_a (torch.Tensor): Appearance embedding for first iceberg
            features_b (torch.Tensor): Appearance embedding for second iceberg

        Returns:
            float | None: Similarity score in [0, 1] if thresholds passed, 
                         None if filtered out
        """
        # Compute raw similarity features
        distance = get_distance(iceberg_a, iceberg_b)
        size_similarity = get_size_similarity(iceberg_a, iceberg_b)
        appearance_similarity = get_appearance_similarity(features_a, features_b, self.device)

        # Normalize distance: convert to similarity (1 = close, 0 = far)
        # Uses robust min-max normalization within threshold range
        distance_norm = 1 - min_max_normalize(
            distance, 0, self.config.thresholds['distance'],
        )

        # Size similarity already in [0, 1] range
        size_norm = size_similarity

        # Appearance similarity already in [0, 1] range (cosine similarity)
        appearance_norm = appearance_similarity

        # Compute weighted score as linear combination
        weighted_score = get_score(
            appearance_norm,
            distance_norm,
            size_norm,
            self.config.weight_appearance,
            self.config.weight_distance,
            self.config.weight_size
        )

        # Apply thresholds with some relaxation (factor of 1.5)
        # This allows matches that are slightly below threshold in one criterion
        # if they're strong in others
        if (distance <= self.config.thresholds['distance'] * 1.5 and
                size_similarity >= self.config.thresholds['size'] / 1.5 and
                appearance_similarity >= self.config.thresholds['appearance'] / 1.5 and
                weighted_score >= self.config.thresholds['match_score'] / 1.5):
            return weighted_score

        return None

    def _matching_bidirectional(self, frame_id, detections, embeddings, spatial_index=None):
        """
        Bidirectional matching algorithm.

        Implements bidirectional matching where a match is only accepted if both
        the track and detection mutually prefer each other. This is more conservative
        than pure greedy matching and can reduce false positives.

        Algorithm:
        Phase 1: For each track, find best detection (track -> detection)
                 For each detection, track who considers it best (detection -> track)
        Phase 2: Accept match only if bidirectional (mutual best match)

        This prevents situations where multiple tracks compete for the same
        detection and ensures more stable associations.

        Args:
            frame_id (int): Current frame ID
            detections (dict): Dictionary of detections {det_id: det_data}
            embeddings (dict): Pre-computed appearance embeddings
            spatial_index (SpatialIndex | None): Spatial index for fast queries

        Returns:
            tuple: (matches, unmatched_tracks, unmatched_dets)
                - matches: List of (track, detection) tuples
                - unmatched_tracks: List of unmatched IcebergTrack objects
                - unmatched_dets: List of unmatched detection dicts
        """
        # Track best match for each track and detection
        track_best_match = {}  # track -> (detection, similarity)
        det_best_match = {}  # det_id -> (track, similarity)

        # Convert detections dict to list for easier iteration
        all_detections = list(detections.values())

        # Phase 1: Find best match for each track
        for track in self.tracks:
            # Get predicted position from Kalman filter
            pred_bbox = track.predicted_bbox
            pred_x, pred_y, pred_w, pred_h = pred_bbox
            pred_center = (pred_x + pred_w / 2, pred_y + pred_h / 2)

            # Determine adaptive search radius based on uncertainty
            if self.config.use_kalman:
                search_radius = track.get_uncertainty()
                # Ensure minimum search radius
                search_radius = max(search_radius, self.config.thresholds['distance'])
            else:
                search_radius = self.config.thresholds['distance']

            # Get candidate detections within search radius
            if spatial_index is not None:
                # Fast spatial query
                candidates = spatial_index.query_radius(pred_center, search_radius)
            else:
                # Fallback: check all detections (slow)
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

                # Skip if embeddings not available
                if track_embedding is None or det_embedding is None:
                    continue

                # Compute similarity score
                track_bbox_dict = {'bbox': track.last_bbox}
                det_bbox_dict = {'bbox': detection['bbox']}

                similarity = self._compute_similarity(
                    track_bbox_dict,
                    det_bbox_dict,
                    track_embedding,
                    det_embedding
                )

                # Update best match if better
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

        # Phase 2: Bidirectional matching - accept only mutual best matches
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

        Collects ALL candidate matches, sorts by similarity (highest first),
        and greedily assigns matches without requiring bidirectional agreement.

        This approach is more permissive than bidirectional matching and can result
        in fewer fragmented tracks. It works particularly well in scenes where
        icebergs move at different speeds or when occlusions create ambiguous
        situations.

        Algorithm:
        1. For each track, find all candidate detections within search radius
        2. Compute similarity for each (track, detection) pair
        3. Sort ALL candidates by similarity (highest first)
        4. Greedily assign matches, processing best matches first
        5. Skip if track or detection already matched

        The key difference from bidirectional is that we consider the GLOBAL
        ranking of all matches rather than just local best matches.

        Args:
            frame_id (int): Current frame ID
            detections (dict): Dictionary of detections {det_id: det_data}
            embeddings (dict): Pre-computed appearance embeddings
            spatial_index (SpatialIndex | None): Spatial index for fast queries

        Returns:
            tuple: (matches, unmatched_tracks, unmatched_dets)
                - matches: List of (track, detection) tuples
                - unmatched_tracks: List of unmatched IcebergTrack objects
                - unmatched_dets: List of unmatched detection dicts
        """
        # Convert detections to list
        all_detections = list(detections.values())

        # Collect ALL candidate matches with their similarities
        all_candidates = []  # List of (similarity, track, detection)

        # For each track, find candidates and compute similarities
        for track in self.tracks:
            # Get predicted position
            pred_bbox = track.predicted_bbox
            pred_x, pred_y, pred_w, pred_h = pred_bbox
            pred_center = (pred_x + pred_w / 2, pred_y + pred_h / 2)

            # Determine adaptive search radius
            if self.config.use_kalman:
                search_radius = track.get_uncertainty()
                search_radius = max(search_radius, self.config.thresholds['distance'])
            else:
                search_radius = self.config.thresholds['distance']

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

                # Compute similarity
                track_bbox_dict = {'bbox': track.last_bbox}
                det_bbox_dict = {'bbox': detection['bbox']}

                similarity = self._compute_similarity(
                    track_bbox_dict,
                    det_bbox_dict,
                    track_embedding,
                    det_embedding
                )

                if similarity is not None:
                    # Store this candidate match
                    all_candidates.append((similarity, track, detection))

        # KEY STEP: Sort ALL candidates by similarity (HIGHEST first)
        # This is what makes pure greedy work better than bidirectional
        # in ambiguous situations - we prioritize the globally best matches
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Greedy assignment: Process best matches first
        matches = []
        matched_tracks = set()
        matched_det_ids = set()

        for similarity, track, detection in all_candidates:
            det_id = detection['id']

            # Skip if already matched (greedy constraint)
            if track in matched_tracks or det_id in matched_det_ids:
                continue

            # Assign this match (no bidirectional check needed!)
            matches.append((track, detection))
            matched_tracks.add(track)
            matched_det_ids.add(det_id)

        # Find unmatched entities
        unmatched_tracks = [t for t in self.tracks if t not in matched_tracks]
        unmatched_dets = [d for d in all_detections if d['id'] not in matched_det_ids]

        return matches, unmatched_tracks, unmatched_dets

    def _save_tracking_results(self, all_results, output_path):
        """
        Save tracking results to file in MOT format.

        Filters results by minimum track length and writes to file in MOTChallenge
        format. Each line represents one detection with format:
        <frame_id>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,1,-1,-1

        The file is then sorted by frame ID for consistency with evaluation tools.

        Args:
            all_results (list): List of tracking result dicts containing:
                - frame_id: Frame number
                - track_id: Track identifier
                - bbox: [x, y, w, h]
                - confidence: Detection confidence
            output_path (Path): Path to save tracking file
        """
        logger.info(f"\nSaving tracking results...")

        # Count detections per track
        track_lengths = defaultdict(int)
        for result in all_results:
            track_lengths[result['track_id']] += 1

        # Filter tracks by minimum length (reduce noise)
        valid_track_ids = {
            tid for tid, length in track_lengths.items()
            if length >= self.config.min_iceberg_id_count
        }

        # Write to file (MOT format)
        with open(output_path, 'w') as f:
            for result in all_results:
                # Skip tracks that are too short
                if result['track_id'] not in valid_track_ids:
                    continue

                # Extract bounding box
                x, y, w, h = result['bbox']

                # Write MOT format line
                # Format: frame,id,x,y,w,h,conf,class,visibility
                f.write(
                    f"{result['frame_id']},{result['track_id']},"
                    f"{x},{y},{w},{h},"
                    f"{result['confidence']},1,-1,-1\n"
                )

        # Sort file by frame ID for consistency
        sort_file(output_path)

        # Log statistics
        filtered_out = len(all_results) - sum(1 for r in all_results if r['track_id'] in valid_track_ids)
        logger.info(f"Total tracking entries: {len(all_results)}")
        logger.info(f"Filtered out (too short): {filtered_out}")
        logger.info(f"Saved to: {output_path}")