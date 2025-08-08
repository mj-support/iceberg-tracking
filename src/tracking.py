import os
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm

from feature_extraction import get_distance, get_size_similarity, get_appearance_similarity, min_max_normalize, \
    get_score, get_gt_thresholds
from utils.helpers import DATA_DIR, load_icebergs_by_frame, sort_file


"""
Iceberg Multi-Object Tracking Pipeline

This module implements a multi-object tracking system for icebergs across 
sequential frames using appearance features, spatial constraints, and 
size-based similarity metrics. The tracker employs a multi-stage filtering 
approach combined with bidirectional matching to create robust iceberg tracks.

Key Components:
- Multi-criteria similarity computation (appearance, distance, size)
- Hierarchical filtering system with configurable thresholds
- Bidirectional matching algorithm for consistent track assignment  
- Pre-computed appearance embeddings integration
- Configurable post-processing and quality filtering
- MOT (Multiple Object Tracking) format output

Algorithm Overview:
1. Load pre-computed appearance embeddings for all detections
2. Compute pairwise similarities between consecutive frames using:
   - Spatial distance constraints
   - Size similarity metrics  
   - Appearance embeddings
3. Perform bidirectional matching to ensure consistent track assignments
4. Post-process results to filter iceberg tracks that do not fulfill requirements
"""


@dataclass
class IcebergTrackingConfig:
    """
    Configuration class to centralize all hyperparameters and settings for iceberg tracking.

    This class contains all the configuration parameters needed for the iceberg tracking
    algorithm, including dataset information, similarity thresholds, weights, and device settings.

    Attributes:
        dataset (str): Name of the dataset to process
        image_format (str): Image file format (default: "JPG")
        seq_length_limit (int | None): Maximum number of frames to process (None for all frames) e.g. first 10
        thresholds (dict): Dictionary containing similarity thresholds for matching
        weight_appearance (float): Weight for appearance similarity in final score
        weight_size (float): Weight for size similarity in final score
        weight_distance (float): Weight for distance similarity in final score
        min_iceberg_id_count (int): The minimum number of frames in which the iceberg appears
        min_iceberg_size (float): Minimum iceberg size in pixels to consider
        device (str): PyTorch device for computations (CPU/GPU)
    """
    # Data configuration
    dataset: str
    image_format: str = "JPG"

    seq_length_limit: int | None = None

    # Minimum thresholds configuration - these are used to filter out poor matches
    thresholds: dict = field(default_factory=lambda: {
        "appearance": 0.4740,  # Minimum appearance similarity threshold
        "distance": 98,  # Maximum distance threshold (pixels)
        "size": 0.5098,  # Minimum size similarity threshold
        "match_score": 0.3799  # Minimum overall match score threshold
    })

    # Weight configuration - used to balance different similarity metrics
    weight_appearance: float = 1.0
    weight_size: float = 1.0
    weight_distance: float = 1.0

    # Quality filters
    min_iceberg_id_count: int = 2  # Minimum detections needed to keep a track
    min_iceberg_size: float = 0.0  # Minimum iceberg size to consider (pixels)

    # Device configuration
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IcebergTracker:
    """
    A class for tracking icebergs across image frames based on detection data.

    This tracker uses a multi-criteria approach combining appearance features,
    spatial distance, and size similarity to match icebergs between consecutive
    frames. The tracking process consists of four main phases:
    1. Load pre-computed embeddings
    2. Compute similarities between detections
    3. Match icebergs using bidirectional matching
    4. Post-process results and filter tracks
    """

    def __init__(self, config: IcebergTrackingConfig):
        """
        Initialize the IcebergTracker with dataset and configuration parameters.

        Args:
            config (IcebergTrackingConfig): Configuration object containing all
                hyperparameters and settings for the tracking algorithm
        """
        self.config = config
        self.dataset = config.dataset
        self.image_format = f".{config.image_format}"

        # Configure file paths based on dataset structure
        base_path = os.path.join(DATA_DIR, self.dataset)
        self.image_dir = os.path.join(base_path, "images", "raw")
        self.detections_file = os.path.join(base_path, "detections", "det.txt")
        self.embeddings_file = os.path.join(base_path, "embeddings", "det_embeddings.pt")
        self.tracking_file = os.path.join(base_path, "results", "mot.txt")

        # Extract threshold values for easy access
        self.appearance_threshold = self.config.thresholds["appearance"]
        self.distance_threshold = self.config.thresholds["distance"]
        self.size_threshold = self.config.thresholds["size"]
        self.match_score_threshold = self.config.thresholds["match_score"]

        self.device = config.device

        # Print configuration summary for user reference
        self._print_configuration()

    def _print_configuration(self):
        """Print a formatted summary of the tracking configuration."""
        print("\n📄 Iceberg Tracking Configuration")
        print("-" * 40)
        print(f"Dataset:                    {self.dataset}")
        print(f"Tracking sequence length:   {self.config.seq_length_limit}")
        print(f"Device:                   {self.device}")
        print("\nSimilarity thresholds of tracked iceberg:")
        print(f"  Appearance:   {self.appearance_threshold:.4f}")
        print(f"  Distance:     {self.distance_threshold:.4f}")
        print(f"  Size:         {self.size_threshold:.4f}")
        print(f"  Match score:  {self.match_score_threshold:.4f}")
        print("\nWeights:")
        print(f"  Appearance: {self.config.weight_appearance}")
        print(f"  Distance:   {self.config.weight_distance}")
        print(f"  Size:       {self.config.weight_size}")
        print("\nIceberg characteristics:")
        print(f"  Each iceberg must occur at least {self.config.min_iceberg_id_count} times.")
        print(f"  The size of each iceberg has to be at least {self.config.min_iceberg_size} pixels.")
        print("-" * 40)

    def track(self):
        """
        Main tracking pipeline that orchestrates the entire tracking process.

        This method executes the four main phases of iceberg tracking:
        1. Load pre-computed embeddings
        2. Compute similarities between icebergs in consecutive frames
        3. Perform bidirectional matching to create tracks
        4. Post-process results and save to file
        """
        # Load iceberg detection data organized by frame
        icebergs_by_frame = load_icebergs_by_frame(self.detections_file)

        # Load pre-computed appearance embeddings
        iceberg_embeddings = self._load_embeddings()

        # Compute forward and backward similarities between frames
        similarities_forward, similarities_backward = self._get_similar_icebergs(icebergs_by_frame, iceberg_embeddings)

        # Match icebergs to create tracks using bidirectional matching
        tracking = self._matching(similarities_forward, similarities_backward)

        # Filter and save final tracking results
        self._postprocessing(tracking, icebergs_by_frame)

        print("\n--- Tracking complete ---")

    def _load_embeddings(self):
        """
        Load pre-computed appearance embeddings for all detected icebergs.

        Returns:
            dict: Dictionary mapping "{frame_id}_{detection_id}" to embedding tensors
        """
        print("\n--- Phase 1: Load embeddings ---")
        print(f"Loading pre-computed embeddings from {self.embeddings_file}...")
        iceberg_embeddings = torch.load(self.embeddings_file)
        print("Embeddings loaded successfully.")
        return iceberg_embeddings

    def _compute_similarity(self, iceberg_a, iceberg_b, features_a, features_b):
        """
        Compute weighted similarity score between two icebergs using multiple criteria.

        The similarity computation follows a hierarchical approach:
        1. Check if spatial distance is within threshold
        2. Check if size similarity is above threshold
        3. Check if appearance similarity is above threshold
        4. Compute overall weighted score

        Args:
            iceberg_a (dict): First iceberg detection data
            iceberg_b (dict): Second iceberg detection data
            features_a (torch.Tensor): Appearance features for first iceberg
            features_b (torch.Tensor): Appearance features for second iceberg

        Returns:
            float | None: Weighted similarity score if all thresholds are met, None otherwise
        """
        # First filter: Check spatial distance between icebergs
        distance = get_distance(iceberg_a, iceberg_b)

        if distance <= self.distance_threshold:
            # Second filter: Check size similarity
            size_similarity = get_size_similarity(iceberg_a, iceberg_b)

            if size_similarity >= self.size_threshold:
                # Third filter: Check appearance similarity
                appearance_similarity = get_appearance_similarity(features_a, features_b, self.device)

                if appearance_similarity >= self.appearance_threshold:
                    # Normalize all similarity metrics to [0, 1] range
                    appearance_similarity_normalized = min_max_normalize(
                        appearance_similarity, self.appearance_threshold, 1
                    )
                    distance_similarity_normalized = 1 - min_max_normalize(
                        distance, 0, self.distance_threshold
                    )
                    size_similarity_normalized = min_max_normalize(
                        size_similarity, self.size_threshold, 1
                    )

                    # Compute unweighted overall score
                    score = get_score(
                        appearance_similarity_normalized,
                        distance_similarity_normalized,
                        size_similarity_normalized
                    )

                    # Final filter: Check if overall score meets threshold
                    if score >= self.match_score_threshold:
                        # Compute final weighted score
                        weighted_score = get_score(
                            appearance_similarity_normalized,
                            distance_similarity_normalized,
                            size_similarity_normalized,
                            self.config.weight_appearance,
                            self.config.weight_distance,
                            self.config.weight_size
                        )

                        return weighted_score

        # Return None if any threshold is not met
        return None

    def _get_similar_icebergs(self, icebergs_by_frame, all_features):
        """
        Compute similarity scores between icebergs in consecutive frames.

        This method creates two dictionaries:
        - similarities_forward: maps frame -> iceberg_id -> list of potential matches in next frame
        - similarities_backward: maps frame -> iceberg_id -> list of potential matches in previous frame

        Args:
            icebergs_by_frame (dict): Dictionary mapping frame IDs to iceberg detections
            all_features (dict): Dictionary mapping detection keys to appearance features

        Returns:
            tuple: (similarities_forward, similarities_backward) dictionaries
        """
        print("\n--- Phase 2: Compute similarities ---")
        print("Starting comparisons: Search for similar icebergs...")

        frames = sorted(icebergs_by_frame.keys())

        # Determine how many frame pairs to process
        if self.config.seq_length_limit is None:
            seq_length = len(frames) - 1
        else:
            seq_length = self.config.seq_length_limit

        # Initialize similarity dictionaries
        # Structure: {frame: {iceberg_id: ([matches], tracking_id)}}
        similarities_forward = {}
        similarities_backward = {}

        # Setup progress bar for frame processing
        progress_bar = tqdm(
            range(seq_length),
            desc="Processing images",
            unit="image",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Main loop: process consecutive frame pairs
        for i in progress_bar:
            f1, f2 = frames[i], frames[i + 1]
            progress_bar.set_description(f"Compute similarities of icebergs from {f1} and {f2}")

            # Get icebergs from current and next frame
            trackers = icebergs_by_frame[f1]  # Icebergs to be tracked from frame f1
            detections = icebergs_by_frame[f2]  # New detections in frame f2

            # Skip if either frame has no detections
            if not trackers or not detections:
                continue

            # Initialize similarity storage for this frame pair
            similarities_forward[f1] = defaultdict(lambda: ([], -1))
            similarities_backward[f2] = defaultdict(lambda: ([], -1))

            # Compare each iceberg in f1 with each iceberg in f2
            for tracker_idx, tracker in enumerate(trackers.values()):
                # Filter out small icebergs
                iceberg_size = tracker["bbox"][2] * tracker["bbox"][3]  # width * height
                if iceberg_size < self.config.min_iceberg_size:
                    continue

                for detection_idx, detection in enumerate(detections.values()):
                    # Fetch pre-computed appearance features
                    features_a = all_features.get(f"{f1}_{tracker["id"]}")
                    features_b = all_features.get(f"{f2}_{detection["id"]}")

                    # Only compute similarity if both features are available
                    if features_a is not None and features_b is not None:
                        similarity_score = self._compute_similarity(tracker, detection, features_a, features_b)

                        # Store mutual matches if similarity is above threshold
                        if similarity_score is not None:
                            forward_entry = (detection["id"], similarity_score)
                            backward_entry = (tracker["id"], similarity_score)
                            similarities_forward[f1][tracker["id"]][0].append(forward_entry)
                            similarities_backward[f2][detection["id"]][0].append(backward_entry)

        # Sort potential matches by similarity score (highest first)
        for frame in similarities_forward:
            icebergs_by_frame = similarities_forward[frame]
            for iceberg in icebergs_by_frame:
                potential_matches = icebergs_by_frame[iceberg][0]
                potential_matches.sort(key=lambda x: x[1], reverse=True)

        for frame in similarities_backward:
            icebergs_by_frame = similarities_backward[frame]
            for iceberg in icebergs_by_frame:
                potential_matches = icebergs_by_frame[iceberg][0]
                potential_matches.sort(key=lambda x: x[1], reverse=True)

        print("Finished comparisons.")
        return similarities_forward, similarities_backward

    def _matching(self, similarities_forward, similarities_backward):
        """
        Perform bidirectional matching to create iceberg tracks.

        This method implements a greedy bidirectional matching algorithm:
        1. For each iceberg in frame f1, find its best match in frame f2
        2. Check if that match in f2 also has the f1 iceberg as its best match
        3. If both directions agree, create/extend a track

        Args:
            similarities_forward (dict): Forward similarities (f1 -> f2)
            similarities_backward (dict): Backward similarities (f2 -> f1)

        Returns:
            list: List of tracking entries (frame, detection_id, track_id)
        """
        print("\n--- Phase 3: Match icebergs ---")
        print("Start matching of similar icebergs...")

        tracking_iter_id = 1  # Global track ID counter
        tracking = []  # List to store all tracking results

        # Process each frame pair
        for frame_iteration, f1 in enumerate(similarities_forward):
            f1_icebergs = similarities_forward[f1]

            # Process each iceberg in the current frame
            for f1_iceberg_id in f1_icebergs:
                potential_f1_matches, f1_tracking_id = f1_icebergs[f1_iceberg_id]

                # Skip if no potential matches found
                if not potential_f1_matches:
                    continue

                # Get the best match for this iceberg in the next frame
                best_f1_match = potential_f1_matches[0][0]

                # Get the corresponding frame f2
                f2 = list(similarities_backward)[frame_iteration]
                f2_icebergs = similarities_backward[f2]

                # Check bidirectional consistency
                potential_f2_matches = f2_icebergs[best_f1_match][0]
                if potential_f2_matches:  # Ensure there are matches
                    best_f2_match = potential_f2_matches[0][0]

                    # Bidirectional match: f1_iceberg -> f2_iceberg -> f1_iceberg
                    if best_f2_match == f1_iceberg_id:
                        # Create new track if this iceberg doesn't have one yet
                        if f1_tracking_id == -1:
                            f1_tracking_id = tracking_iter_id
                            tracking_iter_id += 1
                            # Add the starting point of the track
                            f1_tracking_entry = (f1, f1_iceberg_id, f1_tracking_id)
                            tracking.append(f1_tracking_entry)

                        # Add the matched iceberg to the track
                        f2_tracking_entry = (f2, best_f1_match, f1_tracking_id)
                        tracking.append(f2_tracking_entry)

                        # Propagate track ID for future matching
                        if f2 in similarities_forward:
                            if best_f1_match in similarities_forward[f2]:
                                f2_forward_entry = similarities_forward[f2][best_f1_match]
                                similarities_forward[f2][best_f1_match] = (f2_forward_entry[0], f1_tracking_id)

        print("Finished matching.")
        return tracking

    def _postprocessing(self, tracking, icebergs_by_frame):
        """
        Post-process tracking results and save to file.

        This method:
        1. Filters out tracks with too few detections
        2. Formats results according to MOT (Multiple Object Tracking) format
        3. Saves results to file and sorts them by frame number

        Args:
            tracking (list): List of tracking entries (frame, detection_id, track_id)
            icebergs_by_frame (dict): Original detection data organized by frame
        """
        print("\n--- Phase 4: Postprocessing ---")
        print("Starting postprocessing...")

        tracking_results = []
        min_iceberg_id_count = defaultdict(int)

        # Count occurrences of each track ID
        for frame, det_id, track_id in tracking:
            min_iceberg_id_count[track_id] += 1

        # Identify tracks with insufficient detections
        rare_id = []
        if self.config.min_iceberg_id_count > 2:
            for track_id in min_iceberg_id_count:
                occurrence = min_iceberg_id_count[track_id]
                if occurrence < self.config.min_iceberg_id_count:
                    rare_id.append(track_id)

        # Format tracking results for output (MOT format)
        for frame, det_id, track_id in tracking:
            # Skip tracks with insufficient detections
            if track_id in rare_id:
                continue

            # Get original detection data
            det_data = icebergs_by_frame[frame][det_id]
            left, top, width, height = det_data['bbox']

            # Format: frame,track_id,left,top,width,height,confidence,x,y,z
            tracking_entry = (
                f"{frame},{track_id},{left},{top},{width},{height},"
                f"{det_data['conf']},{det_data['x']},{det_data['y']},{det_data['z']}\n"
            )
            tracking_results.append(tracking_entry)

        # Write results to file
        with open(self.tracking_file, 'w') as f:
            f.writelines(tracking_results)

        # Sort file by frame number for easier analysis
        sort_file(self.tracking_file)
        print("Finished postprocessing.")


# TODO: Implement multiple matching rounds
# For this, remove matched icebergs from forward and backward similarities
# to allow for better matching of remaining icebergs

def main():
    # Configuration parameters
    dataset = "hill_2min_2023-08"
    image_format = "JPG"
    seq_length_limit = 10
    min_iceberg_id_count = 10
    min_iceberg_size = 0

    # Load dataset-specific optimal thresholds
    thresholds = get_gt_thresholds(dataset, image_format)

    # Create configuration object
    config = IcebergTrackingConfig(
        dataset=dataset,
        image_format=image_format,
        seq_length_limit=seq_length_limit,
        min_iceberg_id_count=min_iceberg_id_count,
        min_iceberg_size=min_iceberg_size,
        thresholds=thresholds
    )

    # Initialize and run tracker
    tracker = IcebergTracker(config=config)
    tracker.track()


if __name__ == '__main__':
    main()