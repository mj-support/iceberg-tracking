import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from utils.helpers import DATA_DIR, PROJECT_ROOT, load_icebergs_by_frame, extract_candidates, extract_matches


"""
Iceberg Feature Extraction Module

This module provides functionality for extracting and analyzing similarity 
features of icebergs across images frames (in ground truth data) for 
tracking purposes. It handles statistical thresholds from ground truth data 
in order to use them for tracking. It also handles appearance embeddings 
using deep learning models and comprehensive feature pipeline coordination

Main Components:
- IcebergFeatureExtractor: Main orchestrator class
- Feature extraction functions: distance, appearance, and size similarity
- Statistical analysis and normalization utilities
"""


class IcebergFeatureExtractor:
    """
    Orchestrates extraction of all iceberg features for similarity analysis.

    This class handles the complete pipeline for iceberg feature extraction including:
    - Loading appearance embedding models
    - Computing statistical thresholds from ground truth data
    - Extracting appearance, distance, and size similarity features
    - Normalizing features and computing comprehensive statistics

    The extracted features are used for iceberg tracking across video frames.

    Attributes:
        dataset (str): Name of the dataset being processed
        image_format (str): Format of input images (e.g., "JPG", "PNG")
        print_stats (bool): Whether to print statistics during processing
        image_dir (str): Path to directory containing raw images
        gt_file (str): Path to ground truth annotations file
        embedding_model_path (str): Path to save/load the trained embedding model
        embeddings_path (str): Path to save/load pre-computed embeddings
        device (torch.device): CUDA or CPU device for computation
        output_file (str): Path to save similarity feature statistics
        thresholds (dict): Computed threshold values for tracking
    """

    def __init__(self, dataset, image_format="JPG", print_stats=True):
        """
        Initialize the feature extractor with dataset configuration.

        Args:
            dataset (str): Name of the dataset to process
            image_format (str, optional): Format of input images. Defaults to "JPG".
            print_stats (bool, optional): Whether to print statistics. Defaults to True.
        """
        self.dataset = dataset
        self.image_format = image_format
        self.print_stats = print_stats

        # Set up file paths for different components
        self.image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
        self.gt_file = os.path.join(DATA_DIR, self.dataset, "annotations", "gt.txt")
        self.embedding_model_path = os.path.join(PROJECT_ROOT, "models", "embedding_model.pth")
        self.embeddings_path = os.path.join(DATA_DIR, self.dataset, "embeddings", "gt_embeddings.pt")
        self.output_file = os.path.join(PROJECT_ROOT, "models", "gt_similarity_features.json")

        # Initialize computation device and threshold storage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thresholds = {}

    def generate_similarity_features(self):
        """
        Main pipeline method that orchestrates the complete feature extraction process.

        This method:
        1. Ensures trained embedding model exists
        2. Ensures pre-computed iceberg embeddings exists
        3. Loads embeddings and computes similarity features
        4. Normalizes features and computes statistics
        5. Saves results to JSON file

        Returns:
            dict: Comprehensive statistics about similarity features including
                  raw features, normalized features, and computed thresholds
        """
        # Step 1: Ensure we have a trained embedding model
        if not os.path.exists(self.embedding_model_path):
            raise FileNotFoundError("No embedding model found")

        # Step 2: Ensure we have pre-computed embeddings
        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError("No pre-computed embeddings found")

        # Step 3: Load pre-computed embeddings
        print(f"Loading pre-computed embeddings from {self.embeddings_path}...")
        iceberg_embeddings = torch.load(self.embeddings_path)
        print("Embeddings loaded successfully.")

        # Step 4: Extract similarity features from ground truth data
        icebergs_by_frame = load_icebergs_by_frame(self.gt_file)
        similarity_features = self._get_similarity_features(icebergs_by_frame, iceberg_embeddings)

        # Step 5: Normalize features and compute comprehensive statistics
        normalized_similarity_features = self._normalize_features(similarity_features)
        iceberg_similarity_feature_stats = self._get_similarity_stats(similarity_features,
                                                                     normalized_similarity_features)

        # Step 6: Display and save results
        if self.print_stats:
            self._print_similarity_stats(iceberg_similarity_feature_stats)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(iceberg_similarity_feature_stats, f, ensure_ascii=False, indent=4)
        print(f"\nSimilarity feature stats saved to {self.output_file}")

        return iceberg_similarity_feature_stats

    def _get_similarity_features(self, icebergs_by_frame, iceberg_embeddings):
        """
        Compute similarity features for all matched iceberg pairs in ground truth data.

        For each matched pair of icebergs across consecutive frames, this method computes:
        - Appearance similarity using deep learning embeddings
        - Spatial distance between iceberg centers
        - Size similarity based on bounding box areas

        Args:
            icebergs_by_frame (dict): Dictionary mapping frame names to iceberg detections
            iceberg_embeddings (dict): Pre-computed appearance embeddings for all icebergs

        Returns:
            dict: Dictionary containing lists of similarity values for each feature type:
                  - "appearance": Cosine similarities between embeddings
                  - "distance": Euclidean distances between iceberg centers
                  - "size": Size similarity scores (1 - normalized size difference)
        """
        # Extract ground truth matches from annotation data
        candidates = extract_candidates(self.gt_file)
        matches = extract_matches(candidates)

        # Initialize lists to store similarity measurements
        size_similarities = []
        distances = []
        appearance_similarities = []

        # Process each matched pair
        for match in matches:
            id = match['id']
            frame = match['frame']
            next_frame = match['next_frame']

            # Get iceberg detections for the matched pair
            iceberg_a = icebergs_by_frame[match['frame']][id]
            iceberg_b = icebergs_by_frame[match['next_frame']][id]

            # Compute appearance similarity using pre-computed embeddings
            features_a = iceberg_embeddings.get(f"{frame}_{id}")
            features_b = iceberg_embeddings.get(f"{next_frame}_{id}")
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
        return similarity_features

    def _normalize_features(self, similarity_features):
        """
        Normalize all similarity features to [0,1] range and compute combined scores.

        This method applies appropriate normalization strategies for each feature type:
        - Distance: Inverted and normalized (smaller distances → higher similarity)
        - Size: Normalized using minimum threshold (accounts for size variation tolerance)
        - Appearance: Normalized using minimum threshold
        - Combined score: Weighted average of all normalized features

        Args:
            similarity_features (dict): Raw similarity features from get_similarity_features()

        Returns:
            dict: Normalized features including combined match scores

        Side Effects:
            Updates self.thresholds with computed normalization parameters
        """
        # Normalize distance similarities (invert so closer = more similar)
        distances = similarity_features["distance"]
        distance_threshold = max(distances)  # Use max distance as normalization bound
        distance_similarities_normalized = [1 - min_max_normalize(v, 0, distance_threshold) for v in distances]

        # Normalize size similarities using minimum as threshold
        size_similarities = similarity_features["size"]
        size_threshold = min(size_similarities)  # Minimum acceptable size similarity
        size_similarities_normalized = [min_max_normalize(v, size_threshold, 1) for v in size_similarities]

        # Normalize appearance similarities using minimum as threshold
        appearance_similarities = similarity_features["appearance"]
        appearance_threshold = min(appearance_similarities)  # Minimum acceptable appearance similarity
        appearance_similarities_normalized = [min_max_normalize(v, appearance_threshold, 1) for v in
                                              appearance_similarities]

        # Compute combined match scores as weighted average of all features
        scores = [get_score(a, d, s) for a, d, s in
                  zip(appearance_similarities_normalized, distance_similarities_normalized,
                      size_similarities_normalized)]

        # Store thresholds for future tracking decisions
        self.thresholds = {
            "appearance": appearance_threshold,
            "distance": distance_threshold,
            "size": size_threshold,
            "match_score": float(min(scores))  # Minimum score threshold for valid matches
        }

        # Return all normalized features
        normalized_similarity_features = {
            "appearance": appearance_similarities_normalized,
            "distance": distance_similarities_normalized,
            "size": size_similarities_normalized,
            "match_score": scores,
        }

        return normalized_similarity_features

    def _get_similarity_stats(self, similarity_features, normalized_similarity_features):
        """
        Compute comprehensive statistics for both raw and normalized similarity features.

        Calculates mean, median, standard deviation, min, and max for each feature type
        in both raw and normalized forms. Also includes the computed thresholds.

        Args:
            similarity_features (dict): Raw similarity feature values
            normalized_similarity_features (dict): Normalized similarity feature values

        Returns:
            dict: Comprehensive statistics organized by feature type and normalization status
        """
        # Initialize results dictionary
        similarity_stats = {}

        # Compute statistics for raw similarity features
        similarity_features_stats = {}
        similarity_features_stats["appearance"] = self._calculate_stats(similarity_features["appearance"])
        similarity_features_stats["distance"] = self._calculate_stats(similarity_features["distance"])
        similarity_features_stats["size"] = self._calculate_stats(similarity_features["size"])
        similarity_stats["similarity_features"] = similarity_features_stats

        # Compute statistics for normalized similarity features
        normalized_similarity_features_stats = {}
        normalized_similarity_features_stats["appearance"] = self._calculate_stats(
            normalized_similarity_features["appearance"])
        normalized_similarity_features_stats["distance"] = self._calculate_stats(
            normalized_similarity_features["distance"])
        normalized_similarity_features_stats["size"] = self._calculate_stats(normalized_similarity_features["size"])
        normalized_similarity_features_stats["match_score"] = self._calculate_stats(
            normalized_similarity_features["match_score"])
        similarity_stats["normalized_similarity_features"] = normalized_similarity_features_stats

        # Include computed thresholds
        similarity_stats["thresholds"] = self.thresholds

        return similarity_stats

    def _calculate_stats(self, similarity_values):
        """
        Calculate basic statistical measures for a list of similarity values.

        Args:
            similarity_values (list): List of numerical similarity values

        Returns:
            dict: Dictionary containing mean, median, standard deviation, min, and max
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

        Creates and displays pandas DataFrames showing:
        1. Raw similarity feature statistics
        2. Normalized similarity feature statistics
        3. Computed threshold values for tracking

        Args:
            similarity_stats (dict): Statistics dictionary from get_similarity_stats()
        """
        # Print raw similarity features table
        df = pd.DataFrame(similarity_stats["similarity_features"])
        print("\nSimilarities between matched icebergs in ground truth dataset:")
        print(df.to_string(float_format="%.4f"))

        # Print normalized similarity features table
        df = pd.DataFrame(similarity_stats["normalized_similarity_features"])
        print("\nNormalized similarities between matched icebergs in ground truth dataset:")
        print(df.to_string(float_format="%.4f"))

        # Print threshold values
        thresholds = pd.Series(similarity_stats["thresholds"])
        print("\nThreshold values usable for tracking based on min/max values:")
        print(thresholds.to_string(float_format="%.4f"))


def get_distance(iceberg_a, iceberg_b):
    """
    Calculate Euclidean distance between centers of two iceberg bounding boxes.

    Args:
        iceberg_a (dict): First iceberg with 'bbox' key containing [x, y, w, h]
        iceberg_b (dict): Second iceberg with 'bbox' key containing [x, y, w, h]

    Returns:
        float: Euclidean distance between the centers of the two bounding boxes
    """
    # Extract bounding box coordinates (x, y, width, height)
    a_x, a_y, a_w, a_h = iceberg_a['bbox']
    b_x, b_y, b_w, b_h = iceberg_b['bbox']

    # Calculate Euclidean distance between bounding box centers
    # Note: Using top-left corner coordinates as center approximation
    dist = np.linalg.norm([a_x - b_x, a_y - b_y])
    return dist


def get_appearance_similarity(features_a, features_b, device):
    """
    Compute appearance similarity between two icebergs using cosine similarity of embeddings.

    Takes pre-computed feature vectors and calculates their cosine similarity,
    then scales the result from [-1, 1] range to [0, 1] range for consistency
    with other similarity measures.

    Args:
        features_a (torch.Tensor): Feature embedding for first iceberg
        features_b (torch.Tensor): Feature embedding for second iceberg
        device (torch.device): Device for tensor computation (CPU/CUDA)

    Returns:
        float: Appearance similarity score in [0, 1] range, where 1 is identical
    """
    # Move features to computation device and add batch dimension
    features_a = features_a.to(device).unsqueeze(0)
    features_b = features_b.to(device).unsqueeze(0)

    # Compute cosine similarity between feature vectors (range: [-1, 1])
    cosine_sim = F.cosine_similarity(features_a, features_b, dim=1)

    # Scale similarity from [-1, 1] to [0, 1] for consistent interpretation
    # Formula: (x + 1) / 2 maps [-1, 1] → [0, 1]
    scaled_sim = (cosine_sim + 1) / 2

    # Return as Python float
    appearance_similarity = scaled_sim.item()
    return appearance_similarity


def get_size_similarity(iceberg_a, iceberg_b):
    """
    Calculate size similarity between two icebergs based on bounding box areas.

    Computes the relative difference in areas and converts to a similarity score
    where 1 indicates identical sizes and 0 indicates maximally different sizes.

    Args:
        iceberg_a (dict): First iceberg with 'bbox' key containing [x, y, w, h]
        iceberg_b (dict): Second iceberg with 'bbox' key containing [x, y, w, h]

    Returns:
        float: Size similarity in [0, 1] range, where 1 means identical size
    """
    # Extract bounding box dimensions
    a_x, a_y, a_w, a_h = iceberg_a['bbox']
    b_x, b_y, b_w, b_h = iceberg_b['bbox']

    # Calculate bounding box areas
    size_a = a_w * a_h
    size_b = b_w * b_h

    # Compute size similarity: 1 - (absolute difference / maximum size)
    # This gives 1 for identical sizes and approaches 0 for very different sizes
    size_similarity = 1 - abs(size_a - size_b) / max(size_a, size_b)
    return size_similarity


def get_score(appearance_similarity, distance_similarity, size_similarity,
              appearance_weight=1, distance_weight=1, size_weight=1):
    """
    Compute weighted combined similarity score from individual feature similarities.

    Calculates a weighted average of appearance, distance, and size similarities.
    All input similarities should be in [0, 1] range where 1 indicates high similarity.

    Args:
        appearance_similarity (float): Appearance similarity score [0, 1]
        distance_similarity (float): Distance similarity score [0, 1]
        size_similarity (float): Size similarity score [0, 1]
        appearance_weight (float, optional): Weight for appearance feature. Defaults to 1.
        distance_weight (float, optional): Weight for distance feature. Defaults to 1.
        size_weight (float, optional): Weight for size feature. Defaults to 1.

    Returns:
        float: Combined similarity score [0, 1] where 1 indicates perfect match
    """
    # Calculate weighted average of all similarity components
    total_weight = appearance_weight + distance_weight + size_weight
    score = (appearance_similarity * appearance_weight +
             distance_similarity * distance_weight +
             size_similarity * size_weight) / total_weight
    return score


def min_max_normalize(v, v_min, v_max):
    """
    Apply min-max normalization to scale a value to [0, 1] range.

    Uses the formula: (v - min) / (max - min) to linearly scale
    the input value to the [0, 1] range.

    Args:
        v (float): Value to normalize
        v_min (float): Minimum value in the range
        v_max (float): Maximum value in the range

    Returns:
        float: Normalized value in [0, 1] range
    """
    return (v - v_min) / (v_max - v_min)


def get_gt_thresholds(dataset, image_format):
    """
    Retrieve or compute similarity thresholds from ground truth data for iceberg tracking.

    This function provides a convenient interface to obtain the statistical thresholds
    needed for iceberg tracking decisions. It will either load pre-computed thresholds
    from a cached file or trigger the complete feature extraction pipeline to compute
    them if they don't exist.

    Args:
        dataset (str): Name of the dataset to process (e.g., "hill_2min_2023-08")
        image_format (str): Format of input images (e.g., "JPG", "PNG")

    Returns:
        dict: Dictionary containing threshold values with keys:
            - "appearance" (float): Minimum appearance similarity for valid matches
            - "distance" (float): Maximum distance threshold for spatial consistency
            - "size" (float): Minimum size similarity for valid matches
            - "match_score" (float): Minimum combined score for accepting matches
    """
    # Create feature extractor instance for the specified dataset
    extractor = IcebergFeatureExtractor(dataset, image_format)

    # Check if similarity features have already been computed
    if not os.path.exists(extractor.output_file):
        print("Similarity features not found. Computing features and thresholds...")
        # Run complete feature extraction pipeline to generate thresholds
        extractor.generate_similarity_features()

    # Load pre-computed similarity features and extract thresholds
    print(f"Loading thresholds from {extractor.output_file}...")
    with open(extractor.output_file, 'r', encoding='utf-8') as f:
        similarity_features = json.load(f)
        thresholds = similarity_features['thresholds']

    print("Thresholds loaded successfully.")
    return thresholds

def main():
    # Configuration for the dataset to process
    dataset = "hill_2min_2023-08"
    image_format = "JPG"

    # Create feature extractor and run complete feature extraction pipeline
    extractor = IcebergFeatureExtractor(dataset=dataset, image_format=image_format)
    extractor.generate_similarity_features()


if __name__ == "__main__":
    main()