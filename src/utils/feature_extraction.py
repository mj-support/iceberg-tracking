import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.helpers import load_icebergs_by_frame, extract_candidates, extract_matches, get_sequences

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

def get_gt_thresholds(dataset, print_stats=True):
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
    logger.info("\n=== EXTRACT GROUND TRUTH SIMILARITY FEATURES ===")
    logger.info(f"Processing dataset: {dataset}")

    # Get all sequences in the dataset
    sequences = get_sequences(dataset)
    logger.info(f"Found {len(sequences)} sequences: {list(sequences.keys())}")

    # Initialize aggregation lists for all sequences
    total_similarity_features = {
        "appearance": [],
        "distance": [],
        "distance_normalized": [],
        "size": []
    }

    # Process each sequence independently
    for sequence_name, paths in sequences.items():
        logger.info(f"\nProcessing sequence: {sequence_name}")

        # Step 1: Load pre-computed embeddings
        iceberg_embeddings = torch.load(paths["gt_embeddings"])

        # Step 2: Extract similarity features from ground truth matches
        icebergs_by_frame = load_icebergs_by_frame(paths["ground_truth"])
        similarity_features = get_similarity_features(icebergs_by_frame, iceberg_embeddings, paths["ground_truth"])

        # Step 3: Aggregate features across sequences
        total_similarity_features['appearance'].extend(similarity_features["appearance"])
        total_similarity_features['distance'].extend(similarity_features["distance"])
        total_similarity_features['size'].extend(similarity_features["size"])

    # Step 4: Normalize features and compute comprehensive statistics
    logger.info("\n--- Computing Statistics ---")

    distances = total_similarity_features["distance"]
    normalized_distances = [1 - min_max_normalize(v, 0, max(distances)) for v in distances]
    total_similarity_features["distance_normalized"].extend(normalized_distances)
    similarity_stats = {}

    for feature in total_similarity_features:
        similarity_array = np.array(total_similarity_features[feature])
        similarity_stats[feature] = {
            "Mean": np.mean(similarity_array),
            "Median": np.median(similarity_array),
            "Std Dev": np.std(similarity_array),
            "Min": np.min(similarity_array),
            "Max": np.max(similarity_array),
        }

    # Include computed thresholds in results
    thresholds = {
        "appearance": similarity_stats["appearance"]["Min"],
        "distance": similarity_stats["distance"]["Max"],
        "size": similarity_stats["size"]["Min"]
    }

    # Step 5: Display results if requested
    if print_stats:
        df = pd.DataFrame(similarity_stats)
        logger.info("\n=== SIMILARITIES BETWEEN MATCHED ICEBERGS ===")
        logger.info("\n" + df.to_string(float_format="%.4f"))

        # Print threshold values
        formatted = "\nThresholds values for Tracking:\n" + "\n".join(
            f"  {key}: {float(value):.4f}" for key, value in thresholds.items()
        )
        logger.info(formatted)

    logger.info("\nThresholds loaded successfully.")
    return thresholds


def get_similarity_features(icebergs_by_frame, iceberg_embeddings, ground_truth_file):
    """
    Compute similarity features for all matched iceberg pairs in ground truth.

    This method processes each ground truth match (same iceberg across consecutive
    frames) and computes three types of similarity features. These features capture
    different aspects of iceberg consistency over time.

    Args:
        icebergs_by_frame (dict): Nested dict mapping frame names to iceberg data
            Format: {frame_name: {iceberg_id: {bbox: [x,y,w,h], ...}}}
        iceberg_embeddings (dict): Pre-computed feature vectors
            Format: {"{frame}_{id}": tensor}
        ground_truth_file (str): Path to ground truth annotation file

    Returns:
        dict: Dictionary with three lists of similarity values:
            {
                "appearance": [0.87, 0.91, 0.84, ...],          # Cosine similarities
                "distance": [15.3, 22.7, 8.9, ...],             # Pixel distances
                "size": [0.95, 0.88, 0.97, ...]                 # Size ratios
            }
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
        appearance_similarity = get_appearance_similarity(features_a, features_b, "cpu")
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


def get_score(appearance_similarity, eucl_distance_similarity, kalman_distance_similarity, size_similarity,
              appearance_weight=0.2, eucl_distance_weight=0.2, kalman_distance_weight=0.5, size_weight=0.1):
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
    total_weight = appearance_weight + eucl_distance_weight + kalman_distance_weight + size_weight
    score = (
                    appearance_similarity * appearance_weight +
                    eucl_distance_similarity * eucl_distance_weight +
                    kalman_distance_similarity * kalman_distance_weight +
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