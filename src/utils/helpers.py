import cv2
import os
from collections import defaultdict
from pathlib import Path

"""
Helper Utilities Module

This module provides utility functions and constants for iceberg tracking
and analysis workflows. It handles file parsing, data extraction, image processing,
and coordinate system operations across the tracking pipeline.

Main Components:
- File parsing utilities: annotation loading and sorting functions
- Data extraction functions: candidate matching and iceberg cropping
- Path configuration: project directory structure and constants
- Image processing helpers: bounding box operations and cropping utilities

Features:
- File parsing with error handling for malformed data
- Efficient data structures for frame-based iceberg organization
- Automatic path resolution for cross-platform compatibility
- Comprehensive annotation format support for tracking workflows
"""

# Project directory structure configuration
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # Navigate up two levels from src/tracking/
DATA_DIR = PROJECT_ROOT / "data"  # Main data directory
SRC_DIR = PROJECT_ROOT / "src"  # Source code directory


def sort_file(txt_file):
    """
    Sort annotation file entries by filename and object ID.

    Reads a comma-separated annotation file, sorts entries first by filename
    then by object ID, and writes the sorted data back to the same file.
    Handles malformed data gracefully by assigning infinite ID values.

    Args:
        txt_file (str): Path to the annotation file to be sorted

    Note:
        This function modifies the input file in-place. The expected format
        is: filename,object_id,... (additional fields preserved)
    """
    # Read all lines from the file
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    parsed = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            filename = parts[0]
            try:
                obj_id = int(parts[1])
            except ValueError:
                # Handle malformed object IDs by assigning infinite value
                obj_id = float('inf')  # fallback for bad data
            parsed.append((filename, obj_id, line))

    # Sort by filename first, then by object ID
    parsed.sort(key=lambda x: (x[0], x[1]))
    sorted_lines = [line for _, _, line in parsed]

    # Write sorted data back to file
    with open(txt_file, "w") as f:
        f.writelines(sorted_lines)


def extract_candidates(txt_path):
    """
    Extract iceberg candidates grouped by image frame.

    Parses an annotation file and organizes iceberg IDs by their corresponding
    image frames, creating a dictionary structure for efficient frame-based
    processing.

    Args:
        txt_path (str): Path to the annotation file

    Returns:
        dict: Dictionary mapping image names to lists of iceberg IDs
              Format: {image_name: [id1, id2, ...], ...}
              Results are sorted by image name for consistent ordering
    """
    candidates = defaultdict(list)

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            image_name = parts[0]
            iceberg_id = int(parts[1])
            candidates[image_name].append(iceberg_id)

    # Return sorted dictionary for consistent processing order
    return dict(sorted(candidates.items()))


def extract_matches(candidates):
    """
    Find iceberg matches between consecutive frames.

    Analyzes candidate icebergs across sequential frames to identify which
    icebergs appear in consecutive frames, indicating potential tracking
    continuity.

    Args:
        candidates (dict): Dictionary from extract_candidates() mapping
                          image names to lists of iceberg IDs

    Returns:
        list: List of match dictionaries, each containing:
              - 'id': iceberg ID that appears in consecutive frames
              - 'frame': current frame name
              - 'next_frame': next frame name where ID also appears
    """
    matches = []

    # Iterate through all frames except the last one
    for frame_id, frame in enumerate(candidates):
        if frame_id != len(candidates) - 1:
            current_icebergs = candidates[frame]
            next_frame = list(candidates)[frame_id + 1]
            next_icebergs = candidates[next_frame]

            # Check each iceberg in current frame
            for iceberg in current_icebergs:
                match = {}
                # If iceberg ID exists in next frame, record the match
                if iceberg in next_icebergs:
                    match["id"] = iceberg
                    match["frame"] = frame
                    match["next_frame"] = next_frame
                    matches.append(match)

    return matches


def load_icebergs_by_frame(det_file):
    """
    Load comprehensive iceberg data organized by frame.

    Parses a detection/tracking file containing iceberg information and
    organizes it into a nested dictionary structure for efficient access
    by frame and iceberg ID.

    Args:
        det_file (str): Path to detection/tracking file with format:
                       frame,id,left,top,width,height,conf,x,y,z

    Returns:
        dict: Nested dictionary structure:
              {frame_name: {iceberg_id: {iceberg_data}, ...}, ...}
              Where iceberg_data contains:
              - 'id': integer iceberg ID
              - 'bbox': tuple (left, top, width, height) as floats
              - 'conf': confidence score as float
              - 'x', 'y', 'z': coordinate integers

    Note:
        Results are sorted by frame name for consistent processing order
    """
    icebergs_by_frame = defaultdict(dict)

    with open(det_file, "r") as f:
        for line in f:
            # Parse comma-separated values
            frame, id_, left, top, width, height, conf, x, y, z = line.strip().split(",")

            # Store iceberg data with proper type conversions
            icebergs_by_frame[frame][int(id_)] = {
                'id': int(id_),
                'bbox': (float(left), float(top), float(width), float(height)),
                'conf': float(conf),
                'x': int(x),
                'y': int(y),
                'z': int(z),
            }

    # Return sorted dictionary for consistent frame ordering
    return dict(sorted(icebergs_by_frame.items()))


def parse_annotations(txt_file):
    """
    Parse annotation file into standardized detection format.

    Reads an annotation file and converts it into a list of detection
    dictionaries with standardized field names and data types. Includes
    robust error handling for malformed lines.

    Args:
        txt_file (str): Path to annotation file with format:
                       frame,id,bb_left,bb_top,bb_width,bb_height

    Returns:
        list: List of detection dictionaries, each containing:
              - 'frame': frame identifier (string)
              - 'id': iceberg ID (integer)
              - 'bb_left': left coordinate of bounding box (float)
              - 'bb_top': top coordinate of bounding box (float)
              - 'bb_width': width of bounding box (float)
              - 'bb_height': height of bounding box (float)

    Note:
        - Malformed lines are logged as warnings and skipped
        - Empty lines are automatically ignored
        - Handles both ValueError and IndexError exceptions gracefully
    """
    detections = []

    with open(txt_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue

            parts = line.strip().split(',')
            try:
                # Parse and validate all required fields
                detections.append({
                    'frame': parts[0],
                    'id': int(parts[1]),
                    'bb_left': float(parts[2]),
                    'bb_top': float(parts[3]),
                    'bb_width': float(parts[4]),
                    'bb_height': float(parts[5]),
                })
            except (ValueError, IndexError) as e:
                # Log parsing errors but continue processing
                print(f"Warning: Could not parse line, skipping. Line: '{line.strip()}' | Error: {e}")

    return detections