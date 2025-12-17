import logging
import os
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Helper Utilities Module for Iceberg Tracking System

This module provides essential utility functions and constants for the iceberg tracking
pipeline. It handles file I/O, data parsing, path management, and data structure operations
that are used throughout the detection, tracking, and evaluation workflows.
"""

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

# Project directory structure configuration
# These paths are computed relative to this file's location
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # Navigate up two levels from src/utils/
DATA_DIR = PROJECT_ROOT / "data"  # Main data directory
SRC_DIR = PROJECT_ROOT / "src"  # Source code directory


def parse_cli_args():
    """
    Parse command-line arguments for the iceberg tracking pipeline.

    This function handles the complete CLI argument parsing workflow, including
    command selection, config file specification, and flexible key=value parameter
    overrides. It uses argparse for command structure and custom parsing for
    parameter overrides, providing a clean and intuitive CLI interface.

    Architecture:
        The CLI uses a command-based structure similar to git, docker, or MLflow:

        python run_pipeline.py <command> [cfg=config.yaml] [param=value ...]

    Commands:
        - train-embedding: Train ViT embedding model for appearance similarity
        - train-detection: Train Faster R-CNN detection model
        - detect: Run iceberg detection on images
        - track: Run multi-object tracking
        - eval: Evaluate tracking results against ground truth
        - visualize: Create annotated images and videos

    Parameter Override Syntax:
        Uses simple key=value pairs with automatic type inference:
        - dataset=hill/test          → str
        - max_age=7                  → int
        - conf_threshold=0.7         → float

    Config File Handling:
        Each command has a default config file that can be overridden:
        - cfg=custom.yaml overrides the default
        - If no cfg specified, uses command-specific default
        - Config defaults defined in cfg_defaults dictionary

    Returns:
        tuple: (command, config_file, overrides)
            - command (str): Selected command name
                Examples: 'detect', 'track', 'eval'
            - config_file (str): Path to YAML config file
                Examples: 'cfgs/detect.yaml', 'cfgs/track.yaml'
            - overrides (dict): Key-value parameter overrides
                Examples: {'dataset': 'hill/test', 'max_age': 7}

    Usage Examples:
        >>> # Basic command with default config
        >>> python run_pipeline.py detect dataset=hill/test
        >>> # Returns: ('detect', 'cfgs/detect.yaml', {'dataset': 'hill/test'})

        >>> # Custom config with overrides
        >>> python run_pipeline.py track cfg=cfgs/track.yaml dataset=hill/test max_age=7
        >>> # Returns: ('track', 'cfgs/track_dense.yaml', {'dataset': 'hill/test', 'max_age': 7})

        >>> # Multiple overrides with type inference
        >>> python run_pipeline.py detect dataset=hill/test conf_threshold=0.7 postprocess=false
        >>> # Returns: ('detect', 'cfgs/detect.yaml',
        >>> #          {'dataset': 'hill/test', 'conf_threshold': 0.7, 'use_postprocess': False})

    Notes:
        - The cfg parameter is special: it's extracted from overrides and returned separately
        - All other key=value pairs become config overrides
        - Command is required (parser.add_subparsers with required=True)
        - Unknown arguments without '=' are silently ignored

    Integration:
        This function is called by the python file in the 'script/' directory.
        cmd, cfg_file, overrides = parse_cli_args()
        config = load_config(cfg_file, **overrides)
        # Execute command with config...
    """
    parser = argparse.ArgumentParser(description="Iceberg Pipeline")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # Just define commands, no arguments
    sub.add_parser('train-embedding')
    sub.add_parser('train-detection')
    sub.add_parser('detect')
    sub.add_parser('track')
    sub.add_parser('eval')
    sub.add_parser('visualize')

    args, unknown = parser.parse_known_args()

    # Parse ALL key=value pairs
    overrides = {}
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Type inference
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
            overrides[key] = value

    # Default config files
    cfg_defaults = {
        'train-embedding': 'cfgs/embed.yaml',
        'train-detection': 'cfgs/detect.yaml',
        'detect': 'cfgs/detect.yaml',
        'track': 'cfgs/track.yaml',
        'eval': 'cfgs/eval.yaml',
        'visualize': 'cfgs/visualize.yaml',
    }

    cfg_file = overrides.pop('cfg', cfg_defaults.get(args.cmd))

    return args.cmd, cfg_file, overrides


def load_config(config_file, **overrides):
    """
    Load configuration from YAML file and apply parameter overrides.

    This is the central configuration loading function that handles YAML parsing,
    parameter validation, type conversions, and automatic dataclass instantiation.
    It provides a unified interface for loading configurations in both CLI and
    Python contexts, with support for flexible parameter overrides.

    The function automatically detects the appropriate configuration dataclass
    based on the config filename and returns a properly initialized instance
    with all validations and type conversions applied.

    Architecture:
        YAML File → OmegaConf → Python Dict → Apply Overrides → Dataclass Instance

    Supported Config Types:
        - embed.yaml     → IcebergEmbeddingsConfig
        - detect.yaml    → IcebergDetectionConfig
        - track.yaml     → IcebergTrackingConfig
        - eval.yaml      → EvalConfig
        - visualize.yaml → VisualizationConfig

    Override Priority:
        YAML defaults < Python overrides
        Overrides completely replace YAML values for specified keys.

    Args:
        config_file (str): Path to YAML config file relative to PROJECT_ROOT
            Examples:
                - "cfgs/detect.yaml"
                - "cfgs/track.yaml"
                - "my_experiments/custom_config.yaml"

        **overrides: Keyword arguments to override YAML parameters
            Examples:
                - dataset="hill/test"
                - max_age=7
                - conf_threshold=0.7
                - use_kalman=False

            Any parameter defined in the target dataclass can be overridden.
            Overrides are applied after loading YAML, so they take precedence.

    Returns:
        Dataclass instance: Properly initialized configuration object
            Type depends on config_file name:
            - IcebergEmbeddingsConfig
            - IcebergDetectionConfig
            - IcebergTrackingConfig
            - EvalConfig
            - VisualizationConfig

            All dataclass __post_init__ validations and type conversions
            are automatically applied.

    Raises:
        ValueError: If dataset parameter is missing.

        ValueError: If config_file doesn't match any known config type
            Config filename must contain one of: embed, detect, track, eval, visualize

        FileNotFoundError: If config_file doesn't exist
            Raised by OmegaConf.load()

        TypeError: If overrides contain invalid parameters
            Raised by dataclass constructor

    Usage Examples:
        >>> # Load with default parameters
        >>> config = load_config("cfgs/detect.yaml", dataset="hill/test")
        >>> print(config.conf_threshold)  # From YAML
        0.1

        >>> # Load with overrides
        >>> config = load_config(
        ...     "cfgs/detect.yaml",
        ...     dataset="hill/test",
        ...     conf_threshold=0.7,
        ...     scales=[0.5, 1.0, 1.5]
        ... )
        >>> print(config.conf_threshold)  # Overridden
        0.7

        >>> # From CLI (via parse_cli_args)
        >>> cmd, cfg_file, overrides = parse_cli_args()
        >>> config = load_config(cfg_file, **overrides)

        >>> # In Python script or notebook
        >>> config = load_config("cfgs/track.yaml")
        >>> config.dataset = "hill/test"  # Can also modify after loading
        >>> config.max_age = 7

    Type Detection Logic:
        The function uses simple substring matching to determine config type:
        - "embed" in filename    → IcebergEmbeddingsConfig
        - "detect" in filename   → IcebergDetectionConfig
        - "track" in filename    → IcebergTrackingConfig
        - "eval" in filename     → EvalConfig
        - "visualize" in filename → VisualizationConfig

    Integration:
        This function is the bridge between YAML configs and Python dataclasses:
        a) CLI Path: parse_cli_args() → load_config() → Dataclass → Pipeline
        b) Python Path: load_config() → Dataclass → Pipeline
        c) Notebook Path: load_config() → Modify attributes → Pipeline
    """
    config = OmegaConf.load(PROJECT_ROOT / config_file)
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.update(overrides)

    if not config_dict.get('dataset'):
        raise ValueError(
            f"Dataset path not specified.\n\n"
            f"Provide the dataset (e.g. 'hill/test') using either:\n"
            f"  • Config file in: dataset: 'hill/test' (in cfgs/*.yaml)\n"
            f"  • Command line: dataset=hill/test\n"
            f"  • Python code: config.dataset = 'hill/test'\n\n"
            f"Current value: {config.dataset!r}"
        )

    if "embed" in config_file:
        from embedding import IcebergEmbeddingsConfig
        return IcebergEmbeddingsConfig(**config_dict)
    elif "detect" in config_file:
        from detection import IcebergDetectionConfig
        return IcebergDetectionConfig(**config_dict)
    elif "track" in config_file:
        from tracking import IcebergTrackingConfig
        return IcebergTrackingConfig(**config_dict)
    elif "eval" in config_file:
        from utils.eval import EvalConfig
        return EvalConfig(**config_dict)
    elif "visualize" in config_file:
        from utils.visualize import VisualizationConfig
        return VisualizationConfig(**config_dict)
    else:
        raise ValueError(f"Unknown config type: {config_file}")


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def sort_file(txt_file):
    """
    Sort annotation file entries by frame and object ID in-place.

    This function reads a comma-separated annotation file, sorts all entries
    first by frame number/name, then by object ID, and writes the sorted
    data back to the same file. This standardizes file ordering for consistent
    processing and analysis.

    Sorting Order:
        Primary: Frame number/name (ascending)
        Secondary: Object ID (ascending)

    Args:
        txt_file (str or Path): Path to annotation file to be sorted
    """
    # Read all lines from the file
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Parse each line to extract sortable components
    parsed = []
    for line in lines:
        # Split first field (frame)
        img_name, line_rest = line.split(",", 1)
        # Convert to int if numeric, otherwise keep as string
        img_name = int(img_name) if img_name.isdigit() else img_name

        # Split second field (iceberg ID)
        iceberg_id, line_rest = line_rest.split(",", 1)

        # Reconstruct line with parsed components
        new_line = str(img_name) + "," + iceberg_id + "," + line_rest
        parsed.append((img_name, iceberg_id, new_line))

    # Sort by frame first, then by iceberg ID
    # Both are converted to int for numeric sorting
    parsed.sort(key=lambda x: (int(x[1]), x[0]))
    sorted_lines = [line for _, _, line in parsed]

    # Write sorted data back to file (overwrites original)
    with open(txt_file, "w") as f:
        for line in sorted_lines:
            f.writelines(line)


def parse_annotations(txt_file):
    """
    Parse annotation file into standardized detection dictionaries.

    Reads an annotation file and converts each line into a structured dictionary
    with standardized field names and appropriate data types. Includes robust
    error handling to skip malformed lines while continuing to process valid data.

    This function is used throughout the pipeline to load ground truth, detections,
    and tracking results in a consistent format.

    Args:
        txt_file (str or Path): Path to annotation file

    Returns:
        list: List of detection dictionaries, one per valid line
            Empty list if file is empty or all lines are malformed
    """
    detections = []

    with open(txt_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines
            if not line.strip():
                continue

            # Split comma-separated values
            parts = line.strip().split(',')

            # Format frame identifier (zero-pad if numeric)
            try:
                parts[0] = f"{int(parts[0]):06d}"
            except (ValueError, IndexError):
                # Keep original value if not numeric or missing
                parts[0] = parts[0] if parts else ""

            try:
                # Parse and validate all required fields
                detection = {
                    'frame': parts[0],
                    'id': int(parts[1]),
                    'bb_left': float(parts[2]),
                    'bb_top': float(parts[3]),
                    'bb_width': float(parts[4]),
                    'bb_height': float(parts[5]),
                }
                detections.append(detection)

            except (ValueError, IndexError) as e:
                # Log parsing errors but continue processing
                logger.warning(
                    f"Could not parse line {line_num} in {txt_file}, skipping.\n"
                    f"  Line: '{line.strip()}'\n"
                    f"  Error: {e}"
                )

    return detections


def load_icebergs_by_frame(det_file):
    """
    Load comprehensive iceberg data organized by frame and ID.

    Parses a detection/tracking file and organizes it into a nested dictionary
    structure optimized for frame-by-frame processing. This is the primary
    data loading function used throughout the tracking pipeline.

    Args:
        det_file (str or Path): Path to detection/tracking file

    Returns:
        dict: Nested dictionary with structure:
            {frame_name: {iceberg_id: iceberg_data}}

            Sorted by frame name for consistent iteration
            Empty dict if file is empty or malformed
    """
    icebergs_by_frame = defaultdict(dict)

    with open(det_file, "r") as f:
        for line in f:
            # Parse comma-separated values
            frame, id_, left, top, width, height, conf, x, y, z = line.strip().split(",")

            # Format frame identifier (zero-pad if numeric)
            try:
                frame = f"{int(frame):06d}"
            except (ValueError, TypeError):
                # Keep original value if not numeric
                frame = frame

            # Store iceberg data with proper type conversions
            # Nested dictionary: frame -> iceberg_id -> data
            icebergs_by_frame[frame][int(id_)] = {
                'id': int(id_),
                'bbox': (float(left), float(top), float(width), float(height)),
                'conf': float(conf),
                'x': int(x),
                'y': int(y),
                'z': int(z),
            }

    # Return sorted dictionary for consistent frame ordering
    # Sorting ensures deterministic iteration order
    return dict(sorted(icebergs_by_frame.items()))


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_candidates(txt_path):
    """
    Extract iceberg candidates grouped by frame for matching analysis.

    Parses an annotation file and creates a lightweight mapping from frames
    to lists of iceberg IDs. This is useful for analyzing which icebergs
    appear in which frames without loading full bounding box data.

    Args:
        txt_path (str or Path): Path to annotation file

    Returns:
        dict: Dictionary mapping frame numbers (int) to lists of iceberg IDs (int)
            Keys are sorted for consistent iteration
    """
    candidates = defaultdict(list)

    with open(txt_path, 'r') as f:
        for line in f:
            # Parse comma-separated values
            parts = line.strip().split(",")
            image_name = int(parts[0])  # Frame number
            iceberg_id = int(parts[1])  # Iceberg ID

            # Add iceberg ID to this frame's list
            candidates[image_name].append(iceberg_id)

    # Return as regular dict (sorted by key) for consistent ordering
    return dict(sorted(candidates.items()))


def extract_matches(candidates):
    """
    Find iceberg matches between consecutive frames.

    Analyzes candidate icebergs across sequential frames to identify which
    icebergs appear in consecutive frames. This is useful for validating
    tracking continuity, analyzing track fragmentation, and computing
    frame-to-frame association statistics.

    A "match" occurs when an iceberg ID appears in two consecutive frames,
    suggesting the same physical iceberg was tracked across the frame boundary.

    Args:
        candidates (dict): Dictionary from extract_candidates()
            Format: {frame_num: [iceberg_ids]}

    Returns:
        list: List of match dictionaries, each containing:
            - 'id': Iceberg ID that appears in both frames
            - 'frame': Current frame number
            - 'next_frame': Next frame number
    """
    matches = []

    # Get sorted list of frame numbers
    frame_list = sorted(candidates.keys())

    # Iterate through all frames except the last one
    for frame_idx in range(len(frame_list) - 1):
        current_frame = frame_list[frame_idx]
        next_frame = frame_list[frame_idx + 1]

        current_icebergs = candidates[current_frame]
        next_icebergs = candidates[next_frame]

        # Convert to set for O(1) membership testing
        next_icebergs_set = set(next_icebergs)

        # Check each iceberg in current frame
        for iceberg_id in current_icebergs:
            # If iceberg ID exists in next frame, record the match
            if iceberg_id in next_icebergs_set:
                match = {
                    'id': iceberg_id,
                    'frame': current_frame,
                    'next_frame': next_frame
                }
                matches.append(match)

    return matches


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    IoU is a standard metric measuring overlap between bounding boxes.
    It's the ratio of intersection area to union area.

    Args:
        box1 (list): First bounding box [xmin, ymin, xmax, ymax]
        box2 (list): Second bounding box [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU value in [0, 1]
            - 0.0: No overlap
            - 1.0: Perfect overlap (identical boxes)

    Formula:
        IoU = Intersection Area / Union Area
            = Intersection Area / (Area1 + Area2 - Intersection Area)
    """
    # Find intersection rectangle coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Check if boxes actually intersect
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Calculate areas
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ============================================================================
# PATH MANAGEMENT
# ============================================================================
def get_sequences(dataset):
    """
    Discover and map all sequences in a dataset with their file paths.

    Scans the dataset directory structure and builds a comprehensive mapping
    of all sequences and their associated files (images, annotations, embeddings).
    This is the primary function for dataset discovery and path management.

    The function handles both single-sequence datasets (with images/ at root)
    and multi-sequence datasets (with sequence subdirectories).

    Args:
        dataset (str): Dataset path relative to DATA_DIR
            Examples: "columbia/ice_melange", "columbia/clear"

    Returns:
        dict: Dictionary mapping sequence names to path dictionaries
            {
                "sequence_name": {
                    'images': Path to images directory,
                    'ground_truth': Path to gt.txt,
                    'gt_embeddings': Path to ground truth embeddings,
                    'detections': Path to det.txt,
                    'det_embeddings': Path to detection embeddings,
                    'tracking': Path to track.txt,
                    'track_eval': Path to track_eval.txt,
                    'track_embeddings': Path to tracking embeddings,
                }
            }
    """
    dataset = dataset
    base_path = Path(os.path.join(DATA_DIR, dataset))

    # Determine if single or multi-sequence dataset
    sequences = {}
    images_dir = base_path / 'images'

    if images_dir.exists():
        # Single sequence dataset - images/ at root
        sequence_dirs = [base_path]
    else:
        # Multi-sequence dataset - sequence subdirectories
        sequence_dirs = sorted(base_path.iterdir())

    # Iterate through all subdirectories (sequences)
    for sequence_dir in sequence_dirs:
        # Skip non-directories
        if not sequence_dir.is_dir():
            continue

        sequence_name = sequence_dir.name
        sequence = {}

        # Check for required images directory
        images_dir = sequence_dir / 'images'
        if not images_dir.exists():
            logger.warning(f"No images/ directory in {sequence_dir}, skipping...")
            continue
        sequence['images'] = images_dir

        # Map ground truth paths
        gt_file = sequence_dir / 'ground_truth' / 'gt.txt'
        sequence['ground_truth'] = gt_file

        gt_embeddings_file = sequence_dir / 'ground_truth' / 'embeddings.pt'
        sequence['gt_embeddings'] = gt_embeddings_file

        # Map detection paths
        det_file = sequence_dir / 'detections' / 'det.txt'
        sequence['detections'] = det_file

        det_embeddings_file = sequence_dir / 'detections' / 'embeddings.pt'
        sequence['det_embeddings'] = det_embeddings_file

        # Map tracking paths
        tracking_file = sequence_dir / 'tracking' / 'track.txt'
        sequence['tracking'] = tracking_file

        tracking_eval_file = sequence_dir / 'tracking' / 'track_eval.txt'
        sequence['track_eval'] = tracking_eval_file

        track_embeddings_file = sequence_dir / 'tracking' / 'embeddings.pt'
        sequence['track_embeddings'] = track_embeddings_file

        # Add sequence to collection
        sequences[sequence_name] = sequence

    return sequences


def get_image_ext(image_dir):
    """
    Determine the image file extension used in a directory.

    Samples the first image file in the directory to determine the file
    extension. This allows the pipeline to work with different image formats
    (jpg, png, tiff, etc.) without hardcoding assumptions.

    Args:
        image_dir (str or Path): Path to image directory

    Returns:
        str: File extension without leading dot
            Examples: "jpg", "png", "tiff"
    """
    # Get first file from directory listing
    sample_img = os.listdir(image_dir)[0]

    # Extract extension (everything after last dot)
    sample_ext = sample_img.split('.')[-1]

    return sample_ext