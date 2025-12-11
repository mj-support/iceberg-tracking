from detection import IcebergDetector
from embedding import IcebergEmbeddingsTrainer
from tracking import IcebergTracker
from utils.eval import filter_tracking_to_gt
from utils.visualize import Visualizer
from utils.helpers import parse_cli_args, load_config

"""
Iceberg Detection and Tracking Pipeline

This script provides a unified command-line interface for all iceberg tracking
operations including model training, detection, tracking, evaluation, and
visualization. It uses YAML configuration files with flexible parameter overrides.

Usage:
    python run_pipeline.py <command> [cfg=config.yaml] [param=value ...]

Commands:
    train-embedding    Train ViT embedding model for appearance similarity
    train-detection    Train Faster R-CNN detection model
    detect             Run iceberg detection on images
    track              Run multi-object tracking
    eval               Prepare evaluation of tracking results against ground truth
    visualize          Create annotated images and videos

Examples:
    # Basic detection
    python run_pipeline.py detect dataset=hill/test

    # Tracking with custom config and overrides
    python run_pipeline.py track cfg=cfgs/track.yaml dataset=hill/test max_age=7

    # Visualization with SAM segmentation
    python run_pipeline.py visualize dataset=hill/test draw_contours=true draw_masks=true
"""

def main():
    # Parse command-line arguments
    cmd, cfg_file, overrides = parse_cli_args()

    # ========================================================================
    # TRAINING COMMANDS
    # ========================================================================

    if cmd == 'train-embedding':
        # Train Vision Transformer for appearance-based similarity
        config = load_config(cfg_file, **overrides)
        trainer = IcebergEmbeddingsTrainer(config)
        trainer.run_complete_pipeline()

    elif cmd == 'train-detection':
        # Train Faster R-CNN for iceberg detection
        config = load_config(cfg_file, **overrides)
        detector = IcebergDetector(config)
        detector.train()

    # ========================================================================
    # INFERENCE COMMANDS
    # ========================================================================

    elif cmd == 'detect':
        # Run detection on images
        config = load_config(cfg_file, **overrides)
        detector = IcebergDetector(config)
        detector.predict()

    elif cmd == 'track':
        # Run multi-object tracking
        config = load_config(cfg_file, **overrides)
        tracker = IcebergTracker(config)
        tracker.track()

    # ========================================================================
    # ANALYSIS COMMANDS
    # ========================================================================

    elif cmd == 'eval':
        # Evaluate tracking against ground truth
        config = load_config(cfg_file, **overrides)
        filter_tracking_to_gt(config)

    elif cmd == 'visualize':
        # Create annotated images and videos
        config = load_config(cfg_file, **overrides)
        visualizer = Visualizer(config)
        visualizer.annotate_icebergs()          # Generate annotated images
        # visualizer.render_video()             # Uncomment to also render video


if __name__ == '__main__':
    main()