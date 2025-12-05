import json
import logging
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from utils.helpers import PROJECT_ROOT, load_icebergs_by_frame, extract_candidates, extract_matches, get_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)


"""
Iceberg Feature Extraction and Similarity Analysis Module

This module provides a system for extracting, analyzing, and computing
statistical thresholds for iceberg similarity features. It processes ground truth data
to establish baseline similarity metrics that are used for multi-object tracking decisions.

Purpose:
    The primary goal is to analyze ground truth annotations to understand what constitutes
    a "match" between the same iceberg across frames. By computing statistics on matched
    pairs, we derive thresholds that can be used to make tracking decisions on new data.

Feature Types:
    1. Appearance Similarity: Deep learning-based visual similarity using embeddings
    2. Spatial Distance: Euclidean distance between iceberg centers
    3. Size Similarity: Ratio of bounding box areas

Pipeline Flow:
    1. Load Pre-computed Embeddings: Vision Transformer features for each iceberg
    2. Extract Ground Truth Matches: Pairs of same iceberg in consecutive frames
    3. Compute Similarity Features: Appearance, distance, and size for each pair
    4. Normalize Features: Scale to [0, 1] range for consistent comparison
    5. Compute Statistics: Mean, median, std dev, min, max for each feature
    6. Derive Thresholds: Minimum values define what constitutes a valid match
    7. Save Results: Cache statistics in JSON for tracking algorithms

Key Design Decisions:
    - Min thresholds: Conservative approach using minimum observed values
    - Ensures high recall (captures most true matches)
    - May have lower precision (some false matches)
    - Can be tuned based on application requirements
"""


class IcebergFeatureExtractor:
    """
    Orchestrates extraction and analysis of iceberg similarity features from ground truth.

    This class coordinates the complete pipeline for computing similarity feature statistics
    from ground truth annotations. It analyzes matched iceberg pairs to understand what
    similarity values are typical for correct matches, then derives thresholds for use in
    tracking algorithms.

    The extractor processes multiple sequences, computing appearance, distance, and size
    similarities for all ground truth matches. Statistical analysis yields thresholds that
    balance precision and recall in downstream tracking tasks.

    Key Responsibilities:
        - Load pre-computed appearance embeddings
        - Extract matched pairs from ground truth annotations
        - Compute appearance, distance, and size similarities
        - Normalize features to comparable scales
        - Calculate comprehensive statistics
        - Derive conservative thresholds
        - Save results for tracking algorithms

    Workflow:
        1. Initialization: Set up paths and device
        2. Load embeddings and ground truth
        3. Process each matched pair
        4. Aggregate statistics across sequences
        5. Compute normalization and thresholds
        6. Display and save results

    Attributes:
        dataset (str): Name/path of the dataset being processed
        print_stats (bool): Whether to print statistical tables
        output_file (str): Path to save computed statistics (JSON)
        device (torch.device): CUDA or CPU device for tensor operations
        thresholds (dict): Computed threshold values for tracking decisions

    Methods:
        generate_similarity_features(): Main pipeline coordinator
        _get_similarity_features(): Extract features from matched pairs
        _normalize_features(): Scale features to [0, 1] range
        _get_similarity_stats(): Compute comprehensive statistics
        _calculate_stats(): Calculate mean, median, std, min, max
        _print_similarity_stats(): Display formatted statistics tables
    """

    def __init__(self, dataset, print_stats=True):
        """
        Initialize the feature extractor with dataset configuration.

        Sets up all necessary paths and initializes the computation device.
        The extractor is configured to process a specific dataset and can
        optionally print statistical tables during execution.

        Args:
            dataset (str): Name/path of the dataset to process
                Example: "hill/train"
            print_stats (bool, optional): Whether to display statistics tables.
                Defaults to True for interactive analysis.
        """
        self.dataset = dataset
        self.print_stats = print_stats

        # Set up file path for saving computed statistics
        self.output_file = os.path.join(PROJECT_ROOT, "models", "gt_similarity_features.json")

        # Initialize computation device (prefer GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize threshold storage (populated during feature extraction)
        self.thresholds = {}

        logger.info(f"Initialized IcebergFeatureExtractor for dataset: {dataset}")
        logger.info(f"Using device: {self.device}")

    def generate_similarity_features(self):
        """
        Execute the complete feature extraction and analysis pipeline.

        This is the main entry point that orchestrates all steps of similarity
        feature computation. It processes all sequences in the dataset, computes
        features for all ground truth matches, normalizes the results, calculates
        statistics, and saves everything to a JSON file.

        Pipeline Stages:
            1. Load Pre-computed Embeddings:
            2. Process Ground Truth Matches:
            3. Aggregate Across Sequences:
            4. Normalize Features:
            5. Compute Statistics:
            6. Derive Thresholds:
            7. Display and Save:

        Returns:
            dict: Comprehensive statistics dictionary containing:
                - 'similarity_features': Raw feature statistics
                - 'normalized_similarity_features': Normalized feature stats
                - 'thresholds': Tracking decision thresholds
        """
        logger.info("\n=== SIMILARITY FEATURE EXTRACTION PIPELINE ===")
        logger.info(f"Processing dataset: {self.dataset}")

        # Get all sequences in the dataset
        sequences = get_sequences(self.dataset)
        logger.info(f"Found {len(sequences)} sequences: {list(sequences.keys())}")

        # Initialize aggregation lists for all sequences
        total_similarity_features = {
            "appearance": [],
            "distance": [],
            "size": []
        }

        # Process each sequence independently
        for sequence_name, paths in sequences.items():
            logger.info(f"\n--- Processing sequence: {sequence_name} ---")

            # Step 1: Load pre-computed embeddings
            logger.info(f"Loading pre-computed embeddings from {paths['gt_embeddings']}...")
            iceberg_embeddings = torch.load(paths["gt_embeddings"])
            logger.info(f"Loaded {len(iceberg_embeddings)} embeddings")

            # Step 2: Extract similarity features from ground truth matches
            icebergs_by_frame = load_icebergs_by_frame(paths["ground_truth"])
            similarity_features = self._get_similarity_features(
                icebergs_by_frame,
                iceberg_embeddings,
                paths["ground_truth"]
            )

            # Step 3: Aggregate features across sequences
            total_similarity_features['appearance'].extend(similarity_features["appearance"])
            total_similarity_features['distance'].extend(similarity_features["distance"])
            total_similarity_features['size'].extend(similarity_features["size"])

            logger.info(f"Extracted {len(similarity_features['appearance'])} matched pairs")

        # Step 4: Normalize features and compute comprehensive statistics
        logger.info("\n--- Computing Statistics ---")
        normalized_similarity_features = self._normalize_features(total_similarity_features)
        iceberg_similarity_feature_stats = self._get_similarity_stats(
            total_similarity_features,
            normalized_similarity_features
        )

        # Step 5: Display results if requested
        if self.print_stats:
            self._print_similarity_stats(iceberg_similarity_feature_stats)

        # Step 6: Save results to JSON file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(iceberg_similarity_feature_stats, f, ensure_ascii=False, indent=4)
        logger.info(f"\n✓ Similarity feature statistics saved to {self.output_file}")

        return iceberg_similarity_feature_stats

    def _get_similarity_features(self, icebergs_by_frame, iceberg_embeddings, ground_truth_file):
        """
        Compute similarity features for all matched iceberg pairs in ground truth.

        This method processes each ground truth match (same iceberg across consecutive
        frames) and computes three types of similarity features. These features capture
        different aspects of iceberg consistency over time.

        Feature Computation:
            1. Appearance Similarity:
            2. Spatial Distance:
            3. Size Similarity:

        Args:
            icebergs_by_frame (dict): Nested dict mapping frame names to iceberg data
                Format: {frame_name: {iceberg_id: {bbox: [x,y,w,h], ...}}}
            iceberg_embeddings (dict): Pre-computed feature vectors
                Format: {"{frame}_{id}": tensor}
            ground_truth_file (str): Path to ground truth annotation file

        Returns:
            dict: Dictionary with three lists of similarity values:
                {
                    "appearance": [0.87, 0.91, 0.84, ...],  # Cosine similarities
                    "distance": [15.3, 22.7, 8.9, ...],     # Pixel distances
                    "size": [0.95, 0.88, 0.97, ...]         # Size ratios
                }

        Process:
            1. Extract ground truth matches from annotations
            2. For each matched pair:
               a. Retrieve iceberg detections from both frames
               b. Look up pre-computed embeddings
               c. Compute appearance similarity (cosine)
               d. Compute spatial distance (Euclidean)
               e. Compute size similarity (ratio)
            3. Return organized feature lists
        """
        # Extract ground truth matches from annotation file
        candidates = extract_candidates(ground_truth_file)
        matches = extract_matches(candidates)

        logger.info(f"Processing {len(matches)} ground truth matches...")

        # Initialize lists to store similarity measurements
        size_similarities = []
        distances = []
        appearance_similarities = []

        # Process each matched pair
        for match in matches:
            iceberg_id = match['id']
            frame = match['frame']
            next_frame = match['next_frame']

            # Get iceberg detections for the matched pair
            # Handle both integer and string frame naming
            try:
                iceberg_a = icebergs_by_frame[f"{int(frame):06d}"][iceberg_id]
                iceberg_b = icebergs_by_frame[f"{int(next_frame):06d}"][iceberg_id]
            except:
                iceberg_a = icebergs_by_frame[frame][iceberg_id]
                iceberg_b = icebergs_by_frame[next_frame][iceberg_id]

            # Compute appearance similarity using pre-computed embeddings
            # Embeddings are stored with key format: "{frame}_{iceberg_id}"
            try:
                features_a = iceberg_embeddings.get(f"{int(frame):06d}_{iceberg_id}")
                features_b = iceberg_embeddings.get(f"{int(next_frame):06d}_{iceberg_id}")
            except:
                features_a = iceberg_embeddings.get(f"{frame}_{iceberg_id}")
                features_b = iceberg_embeddings.get(f"{next_frame}_{iceberg_id}")

            # Compute cosine similarity between embeddings
            appearance_similarity = get_appearance_similarity(features_a, features_b, self.device)
            appearance_similarities.append(appearance_similarity)

            # Compute spatial distance between iceberg centers
            distance = get_distance(iceberg_a, iceberg_b)
            distances.append(distance)

            # Compute size similarity based on bounding box areas
            size_similarity = get_size_similarity(iceberg_a, iceberg_b)
            size_similarities.append(size_similarity)

        # Return organized similarity features
        similarity_features = {
            "appearance": appearance_similarities,
            "distance": distances,
            "size": size_similarities,
        }

        logger.info(f"Computed features for {len(appearance_similarities)} pairs")
        return similarity_features

    def _normalize_features(self, similarity_features):
        """
        Normalize all similarity features to [0, 1] range and compute combined scores.

        This method applies appropriate normalization strategies for each feature type
        to ensure they're on comparable scales. It also computes a weighted combined
        score and establishes thresholds for tracking decisions.

        Normalization Strategies:
            1. Distance:
               - Raw values are in pixels (unbounded)
               - Normalized: 1 - (distance / max_distance)
               - Result: Close distances → high similarity (near 1)
               - Far distances → low similarity (near 0)

            2. Size:
               - Already in [0, 1] range (ratio)
               - No transformation needed
               - Direct use in combined score

            3. Appearance:
               - Already in [0, 1] range (cosine scaled)
               - No transformation needed
               - Direct use in combined score

            4. Combined Score:
               - Weighted average of all normalized features
               - Default weights: equal (1, 1, 1)
               - Can be tuned for specific applications

        Args:
            similarity_features (dict): Raw similarity features with keys:
                - 'appearance': List of appearance similarities [0, 1]
                - 'distance': List of distances in pixels
                - 'size': List of size similarities [0, 1]

        Returns:
            dict: Normalized features including combined scores:
                {
                    'appearance': [...],        # Unchanged [0, 1]
                    'distance': [...],          # Inverted and normalized [0, 1]
                    'size': [...],             # Unchanged [0, 1]
                    'match_score': [...]       # Weighted average [0, 1]
                }
        """
        # Normalize distance similarities (invert so closer = more similar)
        # Use max distance as upper bound for normalization
        distances = similarity_features["distance"]
        distance_threshold = max(distances)
        distance_similarities_normalized = [
            1 - min_max_normalize(v, 0, distance_threshold)
            for v in distances
        ]

        logger.info(f"Distance normalization: max = {distance_threshold:.2f} pixels")

        # Compute combined match scores as weighted average of all features
        # Default weights are 1, 1, 1 (equal weighting)
        scores = [
            get_score(a, d, s)
            for a, d, s in zip(
                similarity_features["appearance"],
                distance_similarities_normalized,
                similarity_features["size"]
            )
        ]

        # Store thresholds for future tracking decisions
        # Using minimum values ensures high recall
        self.thresholds = {
            "appearance": min(similarity_features["appearance"]),
            "distance": distance_threshold,
            "size": min(similarity_features["size"]),
            "match_score": float(min(scores))
        }

        logger.info(f"Computed thresholds:")
        logger.info(f"  Appearance: {self.thresholds['appearance']:.4f}")
        logger.info(f"  Distance: {self.thresholds['distance']:.2f} pixels")
        logger.info(f"  Size: {self.thresholds['size']:.4f}")
        logger.info(f"  Match score: {self.thresholds['match_score']:.4f}")

        # Return all normalized features
        normalized_similarity_features = {
            "appearance": similarity_features["appearance"],
            "distance": distance_similarities_normalized,
            "size": similarity_features["size"],
            "match_score": scores,
        }

        return normalized_similarity_features

    def _get_similarity_stats(self, similarity_features, normalized_similarity_features):
        """
        Compute comprehensive statistics for both raw and normalized features.

        Calculates standard statistical measures (mean, median, std dev, min, max)
        for each feature type in both raw and normalized forms. This provides a
        complete picture of the similarity distributions in ground truth data.

        Statistics Interpretation:
            - Mean: Average similarity for true matches (center of distribution)
            - Median: Middle value (robust to outliers)
            - Std Dev: Spread/variability in similarities
            - Min: Conservative threshold (ensures high recall)
            - Max: Upper bound (indicates best-case matches)

        Args:
            similarity_features (dict): Raw similarity values from ground truth
                Keys: 'appearance', 'distance', 'size'
            normalized_similarity_features (dict): Normalized values [0, 1]
                Keys: 'appearance', 'distance', 'size', 'match_score'

        Returns:
            dict: Comprehensive statistics organized hierarchically:
                {
                    'similarity_features': {...},
                    'normalized_similarity_features': {...},
                    'thresholds': {...}
                }

        Process:
            1. Calculate stats for each raw feature
            2. Calculate stats for each normalized feature
            3. Include derived thresholds
            4. Return complete dictionary
        """
        # Initialize results dictionary
        similarity_stats = {}

        # Compute statistics for raw similarity features
        logger.info("Computing raw feature statistics...")
        similarity_features_stats = {}
        similarity_features_stats["appearance"] = self._calculate_stats(similarity_features["appearance"])
        similarity_features_stats["distance"] = self._calculate_stats(similarity_features["distance"])
        similarity_features_stats["size"] = self._calculate_stats(similarity_features["size"])
        similarity_stats["similarity_features"] = similarity_features_stats

        # Compute statistics for normalized similarity features
        logger.info("Computing normalized feature statistics...")
        normalized_similarity_features_stats = {}
        normalized_similarity_features_stats["appearance"] = self._calculate_stats(
            normalized_similarity_features["appearance"]
        )
        normalized_similarity_features_stats["distance"] = self._calculate_stats(
            normalized_similarity_features["distance"]
        )
        normalized_similarity_features_stats["size"] = self._calculate_stats(
            normalized_similarity_features["size"]
        )
        normalized_similarity_features_stats["match_score"] = self._calculate_stats(
            normalized_similarity_features["match_score"]
        )
        similarity_stats["normalized_similarity_features"] = normalized_similarity_features_stats

        # Include computed thresholds in results
        similarity_stats["thresholds"] = self.thresholds

        return similarity_stats

    def _calculate_stats(self, similarity_values):
        """
        Calculate basic statistical measures for a list of similarity values.

        Computes five standard statistical measures that characterize the
        distribution of similarity values in ground truth matches.

        Statistical Measures:
            - Mean: Average value (arithmetic mean)
            - Median: Middle value (50th percentile)
            - Std Dev: Standard deviation (measure of spread)
            - Min: Minimum value (lower bound)
            - Max: Maximum value (upper bound)

        Args:
            similarity_values (list): List of numerical similarity values
                Can be any numeric type (int, float)

        Returns:
            dict: Dictionary containing five statistical measures:
                {
                    'Mean': float,
                    'Median': float,
                    'Std Dev': float,
                    'Min': float,
                    'Max': float
                }
        """
        similarity_array = np.array(similarity_values)
        similarity_stats = {
            "Mean": np.mean(similarity_array),
            "Median": np.median(similarity_array),
            "Std Dev": np.std(similarity_array),
            "Min": np.min(similarity_array),
            "Max": np.max(similarity_array),
        }
        return similarity_stats

    def _print_similarity_stats(self, similarity_stats):
        """
        Print formatted statistics tables for similarity features and thresholds.

        Creates and displays three pandas DataFrames showing comprehensive
        statistics in an easy-to-read tabular format. This is useful for
        interactive analysis and understanding the ground truth distributions.

        Args:
            similarity_stats (dict): Complete statistics dictionary from
                _get_similarity_stats() containing raw features, normalized
                features, and thresholds.
        """
        # Print raw similarity features table
        df = pd.DataFrame(similarity_stats["similarity_features"])
        logger.info("\n" + "=" * 80)
        logger.info("RAW SIMILARITIES BETWEEN MATCHED ICEBERGS (GROUND TRUTH)")
        logger.info("=" * 80)
        logger.info("\n" + df.to_string(float_format="%.4f"))

        # Print normalized similarity features table
        df = pd.DataFrame(similarity_stats["normalized_similarity_features"])
        logger.info("\n" + "=" * 80)
        logger.info("NORMALIZED SIMILARITIES BETWEEN MATCHED ICEBERGS")
        logger.info("=" * 80)
        logger.info("\n" + df.to_string(float_format="%.4f"))

        # Print threshold values
        thresholds = pd.Series(similarity_stats["thresholds"])
        logger.info("\n" + "=" * 80)
        logger.info("THRESHOLD VALUES FOR TRACKING")
        logger.info("=" * 80)
        logger.info("(Based on minimum values from ground truth)")
        logger.info("\n" + thresholds.to_string(float_format="%.4f"))
        logger.info("\n")


# ============================================================================
# SIMILARITY COMPUTATION FUNCTIONS
# ============================================================================

def get_distance(iceberg_a, iceberg_b):
    """
    Calculate Euclidean distance between centers of two iceberg bounding boxes.

    This function measures spatial proximity between two icebergs by computing
    the straight-line distance between their bounding box centers.

    Args:
        iceberg_a (dict): First iceberg with 'bbox' key containing [x, y, w, h]
            where x, y is top-left corner, w, h are width and height
        iceberg_b (dict): Second iceberg with same format

    Returns:
        float: Euclidean distance in pixels between the centers
    """
    # Extract bounding box coordinates (x, y, width, height)
    a_x, a_y, a_w, a_h = iceberg_a['bbox']
    b_x, b_y, b_w, b_h = iceberg_b['bbox']

    # Calculate Euclidean distance between bounding box centers
    dist = np.linalg.norm([
        (a_x + a_w / 2) - (b_x + b_w / 2),  # x-axis difference
        (a_y + a_h / 2) - (b_y + b_h / 2)  # y-axis difference
    ])
    return dist


def get_appearance_similarity(features_a, features_b, device):
    """
    Compute appearance similarity using cosine similarity of deep learning embeddings.

    This function measures visual similarity between two icebergs by comparing
    their pre-computed feature vectors from a Vision Transformer. The cosine
    similarity captures how similar the icebergs appear in terms of texture,
    color, shape, and other visual characteristics.

    Cosine Similarity:
        Measures the angle between two vectors in high-dimensional space.
        Independent of vector magnitude (only direction matters).

        Formula: cos(θ) = (A · B) / (||A|| × ||B||)

    Args:
        features_a (torch.Tensor): Feature embedding for first iceberg
            Shape: [feature_dim] (e.g., 256 dimensions)
        features_b (torch.Tensor): Feature embedding for second iceberg
            Same shape as features_a
        device (torch.device): Device for tensor computation (CPU/CUDA)

    Returns:
        float: Appearance similarity score in [0, 1] range
            1.0 = identical appearance, 0.0 = completely different
    """
    # Move features to computation device and add batch dimension
    # Shape: [feature_dim] → [1, feature_dim]
    features_a = features_a.to(device).unsqueeze(0)
    features_b = features_b.to(device).unsqueeze(0)

    # Compute cosine similarity between feature vectors
    # PyTorch computes: (A · B) / (||A|| × ||B||)
    # Result range: [-1, 1]
    cosine_sim = F.cosine_similarity(features_a, features_b, dim=1)

    # Scale similarity from [-1, 1] to [0, 1] for consistent interpretation
    # Formula: (x + 1) / 2 maps [-1, 1] → [0, 1]
    scaled_sim = (cosine_sim + 1) / 2

    # Return as Python float (remove tensor wrapper)
    appearance_similarity = scaled_sim.item()
    return appearance_similarity


def get_size_similarity(iceberg_a, iceberg_b):
    """
    Compute size similarity using ratio of bounding box areas.

    This function measures how similar two icebergs are in terms of size by
    computing the ratio of their bounding box areas. The ratio approach is
    more intuitive and scale-invariant than absolute differences.

    Formula:
        similarity = min(area_a, area_b) / max(area_a, area_b)

    Args:
        iceberg_a (dict): First iceberg with 'bbox' key [x, y, w, h]
        iceberg_b (dict): Second iceberg with 'bbox' key [x, y, w, h]

    Returns:
        float: Size similarity in [0, 1] range where 1 = identical size
    """
    # Calculate areas from bounding box dimensions
    a_x, a_y, a_w, a_h = iceberg_a['bbox']
    b_x, b_y, b_w, b_h = iceberg_b['bbox']
    size_a = a_w * a_h
    size_b = b_w * b_h

    # Avoid division by zero
    if size_a == 0 or size_b == 0:
        return 0.0

    # Compute ratio of smaller to larger (always ≤ 1.0)
    # This ensures symmetry: ratio(A, B) = ratio(B, A)
    ratio = min(size_a, size_b) / max(size_a, size_b)

    return ratio


def get_score(appearance_similarity, distance_similarity, size_similarity,
              appearance_weight=1, distance_weight=1, size_weight=1):
    """
    Compute weighted combined similarity score from multiple features.

    Args:
        appearance_similarity (float): Visual similarity [0, 1]
        distance_similarity (float): Spatial proximity [0, 1]
        size_similarity (float): Size consistency [0, 1]
        appearance_weight (float, optional): Weight for appearance. Default: 1
        distance_weight (float, optional): Weight for distance. Default: 1
        size_weight (float, optional): Weight for size. Default: 1

    Returns:
        float: Combined similarity score in [0, 1] where 1 = perfect match
    """
    # Calculate weighted average of all similarity components
    total_weight = appearance_weight + distance_weight + size_weight
    score = (
                    appearance_similarity * appearance_weight +
                    distance_similarity * distance_weight +
                    size_similarity * size_weight
            ) / total_weight
    return score


def min_max_normalize(v, v_min, v_max):
    """
    Apply min-max normalization to scale a value to [0, 1] range.

    Min-max normalization is a linear scaling technique that maps values
    from an arbitrary range [v_min, v_max] to the standardized range [0, 1].

    Args:
        v (float): Value to normalize
        v_min (float): Minimum value in the original range
        v_max (float): Maximum value in the original range

    Returns:
        float: Normalized value in [0, 1] range

    """
    return (v - v_min) / (v_max - v_min)


def get_gt_thresholds(dataset):
    """
    Retrieve or compute similarity thresholds from ground truth for tracking.

    This function provides a simple interface to obtain the statistical
    thresholds needed for iceberg tracking decisions. It handles the caching logic:
    - If thresholds already exist, load them from cache
    - If not, run the complete feature extraction pipeline to compute them

    The thresholds represent conservative (minimum) values observed in ground truth
    matches, ensuring high recall in downstream tracking applications.

    Workflow:
        1. Create IcebergFeatureExtractor instance
        2. Check if cached results exist
        3. If not cached:
           a. Load ground truth annotations
           b. Load pre-computed embeddings
           c. Compute similarity features
           d. Normalize and derive thresholds
           e. Save to cache (JSON file)
        4. Load and return thresholds

    Args:
        dataset (str): Name/path of the dataset to process
            Examples: "columbia/clear", "hill_2min_2023-08"

    Returns:
        dict: Threshold values for tracking decisions:
            {
                'appearance': float,     # Min appearance similarity [0, 1]
                'distance': float,       # Max distance (pixels)
                'size': float,          # Min size similarity [0, 1]
                'match_score': float    # Min combined score [0, 1]
            }
    """
    # Create feature extractor instance for the specified dataset
    extractor = IcebergFeatureExtractor(dataset)

    # Check if similarity features have already been computed and cached
    if not os.path.exists(extractor.output_file):
        logger.info("Similarity features not found. Computing features and thresholds...")
        # Run complete feature extraction pipeline to generate thresholds
        extractor.generate_similarity_features()

    # Load pre-computed similarity features and extract thresholds
    logger.info(f"Loading thresholds from {extractor.output_file}...")
    with open(extractor.output_file, 'r', encoding='utf-8') as f:
        similarity_features = json.load(f)
        thresholds = similarity_features['thresholds']

    logger.info("Thresholds loaded successfully.")
    return thresholds