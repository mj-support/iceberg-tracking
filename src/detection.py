import copy
import cv2
import logging
import numpy as np
import pandas as pd
import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dataclasses import dataclass
from datetime import timedelta
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Tuple, List

from embedding import IcebergEmbeddingsConfig, IcebergEmbeddingsTrainer
from utils.helpers import DATA_DIR, PROJECT_ROOT, sort_file, get_sequences, load_icebergs_by_frame, get_image_ext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Iceberg Detection System using Faster R-CNN

This module provides a end-to-end pipeline for detecting icebergs in 
timelapse imagery using deep learning. The system combines Faster R-CNN 
object detection with intelligent postprocessing and embedding generation.

Architecture:
    The pipeline consists of three main components:
    1. Training: K-fold cross-validation with early stopping for robust model selection
    2. Inference: Multi-scale and sliding window detection for comprehensive coverage
    3. Postprocessing: Inline filtering and overlap removal for quality

Key Features:
    - Faster R-CNN with ResNet-50 FPN backbone for accurate detection
    - K-fold cross-validation training for robust model selection
    - Early stopping to prevent overfitting
    - Multi-scale detection for robustness across object sizes
    - Sliding window inference for large high-resolution images
    - Inline filtering during inference (mask, edge, size, confidence)
    - Intelligent overlap removal between detection methods
    - Automatic embedding generation for downstream tracking
    - Nested detection removal to clean up results

Pipeline Flow:
    1. Load multi-sequence dataset with ground truth annotations
    2. Train Faster R-CNN with k-fold cross-validation
    3. For each sequence:
       a. Run multi-scale detection (multiple scales per image)
       b. Run sliding window detection (overlapping windows)
       c. Filter detections inline (confidence, size, edge, mask)
       d. Remove overlaps between detection methods
       e. Remove nested/duplicate detections
    4. Generate appearance embeddings for detected icebergs
    5. Save results in MOTChallenge format for tracking
"""


# ================================
# CONFIGURATION CLASS
# ================================

@dataclass
class IcebergDetectionConfig:
    """
    Centralized configuration class for all iceberg detection hyperparameters and settings.

    This dataclass provides a interface for configuring every aspect
    of the detection pipeline. Using a dataclass ensures all parameters have clear types
    and default values, to configure and maintain the system.

    Categories:
        - Data Configuration: Dataset paths and loading parameters
        - Model Architecture: Network structure and anchor box configuration
        - Training: Optimization and cross-validation settings
        - Detection: Inference behavior and thresholds
        - Inference: Multi-scale and sliding window parameters
        - Postprocessing: Filtering criteria applied inline during inference
        - Hardware: GPU/CPU configuration

    Attributes:
        dataset (str): Name/path of the dataset directory (required)
        num_workers (int): Number of worker processes for data loading parallelization

        # Model Architecture Parameters
        num_classes (int): Number of object classes (background + iceberg = 2)
        anchor_sizes (Tuple): Anchor box sizes for each FPN pyramid level
            Multiple sizes handle objects at different scales
        anchor_aspect_ratios (Tuple): Aspect ratios for anchor boxes at each level
            Common ratios: 0.5 (wide), 1.0 (square), 2.0 (tall)

        # Training Parameters
        k_folds (int): Number of folds for cross-validation (improves generalization)
        num_epochs (int): Maximum training epochs per fold
        batch_size (int): Number of images per training batch
        learning_rate (float): Initial learning rate for SGD optimizer
        momentum (float): SGD momentum parameter (helps escape local minima)
        weight_decay (float): L2 regularization weight (prevents overfitting)

        # Detection Parameters (Faster R-CNN internals)
        box_detections_per_img (int): Maximum detections per image
        box_nms_thresh (float): NMS IoU threshold for final detections
        rpn_nms_thresh (float): NMS IoU threshold for Region Proposal Network
        rpn_score_thresh (float): Minimum score for RPN proposals

        # Inference Parameters
        confidence_threshold (float): Minimum confidence score to keep detections
        scales (List[float]): Scale factors for multi-scale detection
            Example: [0.5, 1.0, 2.0] means test at 50%, 100%, and 200% size
        window_size (Tuple[int, int]): Sliding window dimensions (width, height)
        overlap (float): Overlap ratio between adjacent sliding windows [0, 1)
        iou_threshold (float): IoU threshold for removing duplicate detections

        # Postprocessing Parameters (applied inline during inference)
        postprocess (bool): Whether to apply filtering (typically always True)
        generate_embeddings (bool): Generate appearance embeddings for tracking
        filter_masked_regions (bool): Filter detections in masked areas (e.g., land)
        edge_tolerance (int): Minimum distance from image edge (pixels)
        mask_ratio_threshold (float): Maximum fraction of detection that can be masked
        min_iceberg_size (float): Minimum bounding box area (pixels²)
        max_iceberg_overlap (float): Maximum IoU for nested detection removal

        # Training Checkpoints
        save_checkpoints (bool): Save model weights after each epoch

        # Hardware Configuration
        device (str): PyTorch device ('cuda' or 'cpu')

    Example:
        >>> config = IcebergDetectionConfig(
        ...     dataset="hill/train",
        ...     k_folds=5,
        ...     num_epochs=5,
        ...     confidence_threshold=0.1
        ... )
        >>> detector = IcebergDetector(config)
        >>> detector.train()
        >>> detector.predict()
    """
    # Data configuration
    dataset: str  # Required field
    num_workers: int = 4

    # Model parameters - Define Faster R-CNN architecture
    num_classes: int = 2  # Background + iceberg
    anchor_sizes: Tuple[Tuple[int, ...], ...] = ((16,), (32,), (64,), (128,), (256,))
    anchor_aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 5

    # Training parameters - Control optimization process
    k_folds: int = 5
    num_epochs: int = 4
    batch_size: int = 2
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005

    # Detection parameters - Control model inference behavior
    box_detections_per_img: int = 1000
    box_nms_thresh: float = 0.3
    rpn_nms_thresh: float = 0.5
    rpn_score_thresh: float = 0.0

    # Inference parameters - Control prediction pipeline
    confidence_threshold: float = 0.1
    scales: List[float] = None  # Set in __post_init__
    window_size: Tuple[int, int] = (1024, 1024)
    overlap: float = 0.2
    iou_threshold: float = 0.3

    # Postprocessing parameters - Applied inline during inference
    postprocess: bool = True
    generate_embeddings: bool = True
    edge_tolerance: int = 0
    mask_ratio_threshold: float = 0.02
    filter_masked_regions: bool = True
    min_iceberg_size: float = 0.0
    max_iceberg_overlap: float = 0.5

    # Training checkpoints
    save_checkpoints: bool = False

    # Device configuration
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __post_init__(self):
        """
        Initialize mutable default values after object creation.

        This method is called automatically by dataclass after __init__.
        It's necessary because mutable defaults (like lists) can't be used
        directly in dataclass field definitions.

        Sets default multi-scale factors if not provided.
        """
        if self.scales is None:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


# ================================
# DATASET CLASS
# ================================

class IcebergSequenceDataset(Dataset):
    """
    PyTorch Dataset for a single timelapse sequence of iceberg imagery.

    Each sequence represents a distinct camera viewpoint or environmental condition
    (e.g., 'clear', 'moderate', 'night') with its own images folder and ground truth
    annotations. This class handles loading images and their associated bounding box
    annotations for training.

    Data Format:
        - Images: Image files in {sequence}/images/ directory
        - Ground Truth: CSV file at {sequence}/ground_truth/gt.txt
            Format: frame,iceberg_id,x,y,width,height,conf,x,x,x

    Attributes:
        sequence_name (str): Name of the sequence (e.g., 'clear')
        images_dir (str): Path to the images folder
        transforms (callable): Optional image transforms
        image_ext (str): Image file extension (jpg, png, etc.)
        detections (pd.DataFrame): Ground truth annotations
        unique_images (np.ndarray): List of unique image filenames
        image_to_global_id (dict): Mapping from image name to global unique ID

    Methods:
        __len__(): Returns number of images in sequence
        __getitem__(idx): Returns (image, target) tuple for training
        get_sequence_info(): Returns statistics about the sequence
    """

    def __init__(self, sequence_name, images_dir, gt_file, image_ext, transforms=None):
        """
        Initialize dataset for a single sequence.

        Args:
            sequence_name (str): Name of the sequence (e.g., 'clear', 'moderate')
            images_dir (str): Path to the images folder for this sequence
            gt_file (str): Path to the ground truth annotations file (gt.txt)
            image_ext (str): Image file extension (e.g., 'jpg', 'png')
            transforms (callable, optional): Transform function for data augmentation
        """
        self.sequence_name = sequence_name
        self.images_dir = images_dir
        self.transforms = transforms
        self.image_ext = image_ext

        # Read ground truth annotations
        # Standard MOTChallenge format with some unused columns
        column_names = ['image', 'iceberg_id', 'bb_left', 'bb_top', 'bb_width',
                        'bb_height', 'conf', 'unused_1', 'unused_2', 'unused_3']

        self.detections = pd.read_csv(gt_file, names=column_names)

        # Get unique images in this sequence
        self.unique_images = self.detections['image'].unique()

        # Create mapping for global unique image IDs
        # This ensures same image name from different sequences gets different IDs
        # Important for multi-sequence training where image names might overlap
        self.image_to_global_id = {
            img_name: f"{sequence_name}_{img_name}"
            for img_name in self.unique_images
        }

    def __len__(self):
        """Return the number of images in this sequence."""
        return len(self.unique_images)

    def __getitem__(self, idx):
        """
        Get a single training sample (image and its annotations).

        Args:
            idx (int): Index of the image to retrieve

        Returns:
            tuple: (image_tensor, target_dict) where:
                - image_tensor: [C, H, W] normalized image tensor
                - target_dict: Dictionary with keys:
                    - 'boxes': [N, 4] bounding boxes in (x1, y1, x2, y2) format
                    - 'labels': [N] class labels (all 1 for icebergs)
                    - 'image_id': [1] unique image identifier

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        # Get image name and its annotations
        img_name = self.unique_images[idx]
        img_detections = self.detections[self.detections['image'] == img_name]

        # Try to find image file with various naming conventions
        img_path = None

        # Try with extension directly
        candidate = os.path.join(self.images_dir, f"{img_name}.{self.image_ext}")
        if os.path.exists(candidate):
            img_path = candidate

        # Try with zero-padded format (common for frame sequences)
        if img_path is None:
            candidate = os.path.join(self.images_dir, f"{img_name:06d}.{self.image_ext}")
            if os.path.exists(candidate):
                img_path = candidate

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name} in {self.images_dir}")

        # Load image as PIL Image
        img = Image.open(img_path).convert('RGB')

        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []

        for _, row in img_detections.iterrows():
            # Convert from (x, y, w, h) format to (x1, y1, x2, y2) format
            # This is the format expected by Faster R-CNN
            x1 = row['bb_left']
            y1 = row['bb_top']
            x2 = x1 + row['bb_width']
            y2 = y1 + row['bb_height']

            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # All icebergs are class 1 (0 is background)

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create a unique image_id using sequence name and frame
        # Hash ensures it fits in 32-bit integer range
        global_image_id = hash(f"{self.sequence_name}_{img_name}") % (2 ** 31)

        # Create target dictionary
        # Only include tensor values to avoid device transfer errors during training
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([global_image_id], dtype=torch.int64),
        }

        # Apply transforms if provided
        if self.transforms:
            img = self.transforms(img)
        else:
            # Default: convert to tensor with normalization
            img = F.to_tensor(img)

        return img, target

    def get_sequence_info(self):
        """
        Get statistics about this sequence.

        Returns:
            dict: Dictionary containing:
                - sequence_name: Name of the sequence
                - num_images: Number of unique images
                - num_annotations: Total number of bounding box annotations
                - unique_icebergs: Number of unique iceberg IDs
        """
        return {
            'sequence_name': self.sequence_name,
            'num_images': len(self.unique_images),
            'num_annotations': len(self.detections),
            'unique_icebergs': len(self.detections['iceberg_id'].unique())
        }


# ================================
# MAIN DETECTOR CLASS
# ================================

class IcebergDetector:
    """
    Main iceberg detection class coordinating the complete pipeline.

    This class orchestrates training, inference, and postprocessing for iceberg
    detection across multiple sequences. It implements an optimized pipeline with
    inline filtering.

    Pipeline Overview:
        Training:
            1. Load multi-sequence dataset with ground truth
            2. K-fold cross-validation to find best model
            3. Early stopping to prevent overfitting
            4. Save best model weights

        Inference:
            1. Load trained model
            2. For each sequence and image:
               a. Multi-scale detection (test at multiple scales)
               b. Sliding window detection (process large images)
               c. Inline filtering (confidence, size, edge, mask)
               d. Intelligent overlap removal
            3. Remove nested/duplicate detections
            4. Generate appearance embeddings

    Performance Optimizations:
        - All filtering happens during inference (no postprocessing I/O)
        - Early rejection of invalid detections before NMS

    Attributes:
        config (IcebergDetectionConfig): Configuration object
        dataset (str): Dataset name/path
        device (torch.device): Computing device (GPU/CPU)
        model (torch.nn.Module): Faster R-CNN model
        base_path (str): Base directory for dataset
        model_file (str): Path to save/load model weights
        checkpoint_dir (str): Directory for training checkpoints

    Methods:
        train(): Train model with k-fold cross-validation
        predict(): Run inference on all sequences
        test_all_checkpoints(): Compare results from different checkpoints
    """

    def __init__(self, config: IcebergDetectionConfig):
        """
        Initialize the iceberg detector.

        Sets up all necessary directories and file paths based on the dataset
        configuration. Creates output directories if they don't exist.

        Args:
            config (IcebergDetectionConfig): Configuration object with all parameters
        """
        self.config = config
        self.dataset = config.dataset
        self.device = config.device if config.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        # Create necessary output directories
        self.base_path = os.path.join(DATA_DIR, self.dataset)
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

        # Define file paths for model storage
        self.model_file = os.path.join(PROJECT_ROOT, "models", "iceberg_detection_model.pth")
        self.iceberg_embedding_model = os.path.join(PROJECT_ROOT, "models", "iceberg_embedding_model.pth")
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints")

        # Setup checkpoint directory if enabled
        if self.config.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        logger.info(f"Initialized IcebergDetector for dataset: {self.dataset}")
        logger.info(f"Device: {self.device}")

    def train(self):
        """
        Train the Faster R-CNN model with k-fold cross-validation.

        Implements a comprehensive training pipeline with robust practices:
        - K-fold cross-validation for reliable model selection
        - Early stopping to prevent overfitting
        - SGD optimizer with momentum for stable training
        - Progress tracking with time estimates
        - Checkpoint saving for each epoch (optional)

        The method trains multiple models (one per fold), tracks validation
        loss for each, and saves the best performing model overall.

        Training Process:
            1. Load multi-sequence dataset
            2. For each fold:
               a. Split into train/validation sets
               b. Train for num_epochs (or until early stopping)
               c. Track best validation loss
               d. Optionally save checkpoints
            3. Save overall best model
        """
        # Extract configuration parameters
        k_folds = self.config.k_folds
        num_epochs = self.config.num_epochs
        batch_size = self.config.batch_size
        learning_rate = self.config.learning_rate

        logger.info("\n=== TRAINING PHASE ===")
        logger.info(f"\nStarting training with {k_folds}-fold cross-validation")

        # Load combined dataset from all sequences
        dataset = self._get_multi_seq_dataset()

        # Initialize cross-validation and tracking variables
        total_start_time = time.time()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_val_loss_overall = float('inf')
        best_model_state_overall = None

        # Cross-validation training loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            logger.info(f"\nFold {fold + 1}/{k_folds}")

            # Create data subsets and loaders for this fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Create data loaders with custom collate function
            # collate_fn handles variable number of objects per image
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=self.config.num_workers, collate_fn=lambda x: tuple(zip(*x))
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=self.config.num_workers, collate_fn=lambda x: tuple(zip(*x))
            )

            # Initialize fresh model and optimizer for this fold
            model = self._build_model()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )

            # Training variables for this fold
            best_val_loss = float('inf')
            best_model_state = None

            # Epoch training loop
            for epoch in range(num_epochs):
                # Train one epoch and evaluate
                train_loss = self._train_one_epoch(model, optimizer, train_loader)
                val_loss = self._evaluate_model(model, val_loader)

                # Calculate progress and time estimates
                current_time, estimated_remaining, avg_time = self._calculate_time_estimates(
                    total_start_time, k_folds, fold, num_epochs, epoch
                )

                logger.info(f"Epoch [{epoch + 1}/{num_epochs}] "
                            f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                            f"Time: {timedelta(seconds=int(current_time))}<"
                            f"{timedelta(seconds=int(estimated_remaining))}, "
                            f"{timedelta(seconds=int(avg_time))}/Epoch")

                # Save checkpoint after each epoch if enabled
                if self.config.save_checkpoints:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"model_fold{fold + 1}_epoch{epoch + 1}_valloss{val_loss:.4f}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"  → Checkpoint saved: {os.path.basename(checkpoint_path)}")

                # Check for improvement and update best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

            # Update overall best model across all folds
            if best_val_loss < best_val_loss_overall:
                best_val_loss_overall = best_val_loss
                best_model_state_overall = best_model_state
                logger.info(f"New best model with validation loss: {best_val_loss_overall:.4f}")

        # Save the best model found across all folds
        if best_model_state_overall:
            torch.save(best_model_state_overall, self.model_file)
            logger.info(f"\nBest model saved with validation loss: {best_val_loss_overall:.4f}")
            logger.info(f"Training completed in {timedelta(seconds=int(time.time() - total_start_time))}")

    def predict(self, output_subdir=None, checkpoint_path=None):
        """
        Run inference with inline filtering.

        This method implements a detection pipeline that applies all
        filtering criteria during inference.

        Pipeline:
            1. Load trained model
            2. Load mask for filtering (if enabled)
            3. For each sequence and image:
               a. Multi-scale detection with immediate filtering
               b. Sliding window detection with immediate filtering
               c. Intelligent overlap removal
               d. Save results directly (no postprocessing)
            4. Remove nested detections
            5. Generate embeddings (if enabled)

        Args:
            output_subdir (str, optional): Custom subdirectory for saving results.
                If None, saves to default 'detections' directory.
                If provided, saves to 'detections_{output_subdir}'.
                Useful for comparing results from different checkpoints.
            checkpoint_path (str, optional): Path to specific checkpoint to load.
                If None, loads the default best model.

        Example:
            # Use default best model
            detector.predict()

            # Compare different checkpoints
            detector.predict(output_subdir="epoch3",
                           checkpoint_path="models/checkpoints/model_fold1_epoch3.pth")
        """
        confidence_threshold = self.config.confidence_threshold

        logger.info("\n=== INFERENCE PHASE ===")
        logger.info("Starting prediction with inline filtering...")

        # Load model
        self._load_model(checkpoint_path)
        self.model.eval()

        # Load mask and get image dimensions if filtering is enabled
        mask = None
        img_width, img_height = None, None

        if self.config.filter_masked_regions or self.config.edge_tolerance > 0:
            # Get image dimensions from first available sequence
            sequences = get_sequences(self.dataset)
            first_seq = next(iter(sequences.values()))
            first_seq_image_dir = first_seq["images"]
            image_ext = get_image_ext(first_seq_image_dir)
            sample_img = [f for f in os.listdir(first_seq_image_dir)
                          if f.endswith(image_ext)][0]
            sample_path = os.path.join(first_seq_image_dir, sample_img)
            img_height, img_width = cv2.imread(sample_path).shape[:2]

            # Load mask if masking is enabled
            if self.config.filter_masked_regions:
                mask_file = None
                mask_dir = os.path.join(DATA_DIR, self.dataset.split("/")[0])

                # Try multiple image extensions
                for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                    candidate = os.path.join(mask_dir, f"mask{ext}")
                    if os.path.exists(candidate):
                        mask_file = candidate
                        break

                if mask_file is None:
                    logger.warning(f"Mask file not found in {mask_dir}")
                else:
                    # Load mask as binary (True = masked/land, False = valid/water)
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(bool)
                    logger.info(f"Loaded mask: {img_width}x{img_height}")
                    logger.info(f"Mask filtering enabled (threshold: {self.config.mask_ratio_threshold})")

        # Log active filtering configuration
        if self.config.edge_tolerance > 0:
            logger.info(f"Edge filtering enabled (tolerance: {self.config.edge_tolerance}px)")
        if self.config.min_iceberg_size > 0:
            logger.info(f"Size filtering enabled (min size: {self.config.min_iceberg_size}px²)")
        if self.config.confidence_threshold > 0:
            logger.info(f"Confidence filtering enabled (threshold: {self.config.confidence_threshold})")

        # Process all sequences
        logger.info(f"\nLoading {self.dataset} sequences from: {DATA_DIR}")
        sequences = get_sequences(self.dataset)
        logger.info(f"Total sequences: {len(sequences)}")
        logger.info(f"Sequence names: {', '.join(sequences.keys())}")

        for sequence_name, paths in sequences.items():
            # Override detection path if custom output subdirectory is specified
            if output_subdir is not None:
                # Create custom detections directory
                custom_det_dir = paths['detections'].parent.parent / f'detections_{output_subdir}'
                custom_det_dir.mkdir(parents=True, exist_ok=True)
                custom_det_file = custom_det_dir / 'det.txt'
                paths['detections'] = custom_det_file
                logger.info(f"Saving detections to: {custom_det_file}")
            else:
                # Use default detections directory
                base_path = str(paths["images"]).split("/images")[0]
                det_dir = os.path.join(base_path, "detections")
                os.makedirs(det_dir, exist_ok=True)

            # Get all image files in sequence
            image_ext = get_image_ext(paths['images'])
            image_files = [f for f in os.listdir(paths["images"])
                           if f.endswith(image_ext)]
            all_detections = []
            total_start_time = time.time()

            logger.info(f"\nProcessing {len(image_files)} images in {sequence_name} sequence...")

            # Process each image
            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(paths["images"], img_file)
                img_name = os.path.splitext(img_file)[0]

                # Run both detection methods with inline filtering
                multi_scale_dets = self._run_multi_scale_prediction(
                    img_path, confidence_threshold, mask, img_width, img_height
                )
                sliding_window_dets = self._run_sliding_window_prediction(
                    img_path, confidence_threshold, mask, img_width, img_height
                )

                # Combine detections (already filtered, so much faster!)
                final_detections = self._remove_overlaps(
                    multi_scale_dets, sliding_window_dets, self.config.iou_threshold
                )

                # Add metadata for output
                for i, det in enumerate(final_detections):
                    det['image'] = img_name
                    det['object_id'] = i + 1
                    all_detections.append(det)

                # Progress logging every 10 images
                if (idx + 1) % 10 == 0 or idx == len(image_files) - 1:
                    logger.info(f"  Processed {idx + 1}/{len(image_files)} images, "
                                f"{len(all_detections)} detections so far...")

            detections_len = len(all_detections)

            # Save detections (already filtered, no postprocessing needed!)
            self._save_detections(all_detections, paths['detections'])

            # Remove nested detections if overlap threshold is set
            if self.config.max_iceberg_overlap < 1.0:
                removed_detections = self._remove_nested_detections(paths['detections'])
                detections_len -= removed_detections

            logger.info(f"\n✓ {sequence_name}: {detections_len} detections across "
                        f"{len(image_files)} images")
            logger.info(f"  Detections saved to: {paths['detections']}")
            logger.info(f"  Inference time: {timedelta(seconds=int(time.time() - total_start_time))}")

        # Generate embeddings if enabled
        if self.config.generate_embeddings:
            logger.info("\n=== GENERATING EMBEDDINGS ===")
            total_start_time = time.time()
            config = IcebergEmbeddingsConfig(dataset=self.dataset)
            trainer = IcebergEmbeddingsTrainer(config)
            trainer.generate_iceberg_embeddings("detections")
            logger.info(f"Embeddings generated in {timedelta(seconds=int(time.time() - total_start_time))}")

        logger.info("\n=== COMPLETED ===")

    # ================================
    # MODEL BUILDING METHODS
    # ================================

    def _get_multi_seq_dataset(self):
        """
        Load and combine datasets from all sequences.

        Creates a unified dataset by loading each sequence's images and ground
        truth annotations, then concatenating them. This allows training on
        diverse data from multiple environmental conditions simultaneously.

        Returns:
            ConcatDataset: Combined dataset from all sequences

        Raises:
            ValueError: If no valid sequences found

        The method:
            1. Iterates through all sequences in the dataset
            2. Creates an IcebergSequenceDataset for each valid sequence
            3. Combines all sequences into a single ConcatDataset
            4. Logs statistics about the combined dataset
        """
        transforms = self._get_transforms()
        sequence_datasets = []
        sequence_names = []

        logger.info(f"\nLoading {self.dataset} sequences from: {DATA_DIR}")
        sequences = get_sequences(self.dataset)

        for sequence_name, paths in sequences.items():
            # Check if ground truth exists
            if not paths["ground_truth"].exists():
                logger.info(f"⚠ Warning: No gt.txt found at {paths['ground_truth']}, skipping...")
                continue

            image_ext = get_image_ext(paths["images"])

            # Create dataset for this sequence
            try:
                seq_dataset = IcebergSequenceDataset(
                    sequence_name=sequence_name,
                    images_dir=str(paths["images"]),
                    gt_file=str(paths["ground_truth"]),
                    transforms=transforms,
                    image_ext=image_ext
                )

                sequence_datasets.append(seq_dataset)
                sequence_names.append(sequence_name)

                # Log sequence info
                info = seq_dataset.get_sequence_info()
                logger.info(f"✓ Loaded '{sequence_name}': "
                            f"{info['num_images']} images, "
                            f"{info['num_annotations']} annotations, "
                            f"{info['unique_icebergs']} unique icebergs")

            except Exception as e:
                logger.error(f"✗ Error loading sequence '{sequence_name}': {e}")
                continue

        if not sequence_datasets:
            raise ValueError(f"No valid sequences found in {self.dataset}")

        # Combine all sequence datasets
        combined_dataset = ConcatDataset(sequence_datasets)

        # Print summary statistics
        total_images = len(combined_dataset)
        total_annotations = sum(
            ds.get_sequence_info()['num_annotations']
            for ds in sequence_datasets
        )
        total_unique_icebergs = sum(
            ds.get_sequence_info()['unique_icebergs']
            for ds in sequence_datasets
        )

        logger.info(f"\n{self.dataset} dataset summary:")
        logger.info(f"  Total sequences: {len(sequence_datasets)}")
        logger.info(f"  Sequences: {', '.join(sequence_names)}")
        logger.info(f"  Total images: {total_images}")
        logger.info(f"  Total annotations: {total_annotations}")
        logger.info(f"  Total unique icebergs: {total_unique_icebergs}")
        logger.info("")

        return combined_dataset

    def _build_model(self):
        """
        Build Faster R-CNN model with custom configuration.

        Creates a Faster R-CNN object detection model with:
        - ResNet-50 Feature Pyramid Network (FPN) backbone
        - Custom anchor generator matching configuration
        - Modified classifier head for iceberg detection

        Returns:
            torch.nn.Module: Configured Faster R-CNN model on appropriate device

        Architecture Details:
            Backbone: ResNet-50 with FPN for multi-scale features
            RPN: Region Proposal Network with custom anchors
            ROI Head: Classification and bounding box regression

        The model uses pretrained ImageNet weights for the backbone,
        then replaces the final classification layer for our task.
        """
        # Create custom anchor generator with specified sizes and aspect ratios
        # Multiple sizes help detect objects at different scales
        # Multiple aspect ratios handle different iceberg shapes
        anchor_generator = AnchorGenerator(
            sizes=self.config.anchor_sizes,
            aspect_ratios=self.config.anchor_aspect_ratios
        )

        # Build Faster R-CNN model with custom parameters
        model = fasterrcnn_resnet50_fpn(
            weights=None,  # Don't load full pretrained model (loaded separately)
            rpn_anchor_generator=anchor_generator,
            box_detections_per_img=self.config.box_detections_per_img,
            box_nms_thresh=self.config.box_nms_thresh,
            rpn_nms_thresh=self.config.rpn_nms_thresh,
            rpn_score_thresh=self.config.rpn_score_thresh,
        )

        # Load compatible pretrained weights for faster convergence
        self._load_pretrained_weights(model)

        # Replace classifier head for our specific number of classes
        # Original head is for 1000 ImageNet classes, we need just 2 (background + iceberg)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

        return model.to(self.device)

    def _load_pretrained_weights(self, model):
        """
        Load compatible pretrained weights while skipping incompatible layers.

        Carefully loads pretrained ImageNet weights for the backbone and
        compatible parts of the model, while avoiding shape mismatches that
        can occur with custom anchor configurations.

        Args:
            model (torch.nn.Module): Model to load weights into

        This selective loading ensures we benefit from pretrained features
        while accommodating our custom configuration.
        """
        # Get default pretrained weights
        pretrained_dict = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict()
        model_dict = model.state_dict()

        # Filter out keys with shape mismatches
        # This happens when anchor sizes or class counts differ
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        # Update model dictionary with compatible pretrained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def _load_model(self, checkpoint_path=None):
        """
        Load trained model from saved weights file.

        Args:
            checkpoint_path (str, optional): Path to specific checkpoint.
                If None, loads the default best model.

        Returns:
            torch.nn.Module: Loaded model in evaluation mode

        Raises:
            FileNotFoundError: If model file doesn't exist

        The model architecture must match the saved weights.
        """
        if self.model is None:
            self.model = self._build_model()

        # Use checkpoint path if provided, otherwise use default model
        file_path = self.model_file if checkpoint_path is None else checkpoint_path

        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model loaded from {file_path}")
        else:
            raise FileNotFoundError(f"No trained model found at {file_path}")

        return self.model

    def test_all_checkpoints(self, checkpoint_pattern="model_fold1_epoch*"):
        """
        Test all checkpoints matching a pattern and compare results.

        Useful for finding the optimal epoch that balances underfitting and
        overfitting. Tests each checkpoint and reports detection statistics.

        Args:
            checkpoint_pattern (str): Glob pattern for checkpoint files.
                Default: "model_fold1_epoch*" (all epochs from fold 1)

        Returns:
            list: List of dicts with results for each checkpoint:
                - checkpoint: Checkpoint filename
                - epoch: Epoch number
                - total_detections: Total number of detections
                - total_images: Number of images processed
                - avg_per_image: Average detections per image

        Example:
            >>> results = detector.test_all_checkpoints("model_fold1_epoch*")
            >>> # Compare which epoch gives best detection count
        """
        import glob

        # Find all matching checkpoints
        pattern = os.path.join(self.checkpoint_dir, checkpoint_pattern + ".pth")
        checkpoints = sorted(glob.glob(pattern))

        if not checkpoints:
            logger.warning(f"No checkpoints found matching: {pattern}")
            return {}

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TESTING {len(checkpoints)} CHECKPOINTS")
        logger.info(f"{'=' * 60}\n")

        results = []

        for checkpoint_path in checkpoints:
            checkpoint_name = os.path.basename(checkpoint_path)

            # Extract epoch number from filename
            epoch_match = checkpoint_name.split('epoch')[1].split('_')[0]
            epoch = int(epoch_match)

            logger.info(f"\n--- Testing: {checkpoint_name} ---")

            # Run prediction to custom subdirectory
            output_subdir = f"epoch{epoch}_test"
            self.predict(output_subdir=output_subdir, checkpoint_path=checkpoint_path)

            # Count detections in results
            sequences = get_sequences(self.dataset)
            total_detections = 0
            total_images = 0

            for sequence_name, paths in sequences.items():
                det_file = paths['detections'].parent.parent / f'detections_{output_subdir}' / 'det.txt'

                if det_file.exists():
                    with open(det_file, 'r') as f:
                        lines = f.readlines()
                        total_detections += len(lines)

                    # Count unique images
                    images = set(line.split(',')[0] for line in lines)
                    total_images += len(images)

            avg_detections = total_detections / total_images if total_images > 0 else 0

            results.append({
                'checkpoint': checkpoint_name,
                'epoch': epoch,
                'total_detections': total_detections,
                'total_images': total_images,
                'avg_per_image': avg_detections
            })

            logger.info(f"  Total detections: {total_detections}")
            logger.info(f"  Avg per image: {avg_detections:.1f}")

        # Print summary table
        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY OF ALL EPOCHS")
        logger.info(f"{'=' * 60}")
        logger.info(f"{'Epoch':<8} {'Avg Detections/Image':<25} {'Total Detections':<20}")
        logger.info("-" * 60)

        for r in results:
            logger.info(f"{r['epoch']:<8} {r['avg_per_image']:<25.1f} {r['total_detections']:<20}")

        return results

    # ================================
    # TRAINING METHODS
    # ================================

    def _get_transforms(self):
        """
        Get image transforms for training/inference.

        Returns:
            callable: Transform function that converts images to tensors

        Creates a flexible transform function that handles both PIL Images
        and numpy arrays, normalizing pixel values to [0, 1] range.
        """

        def transform(image):
            if isinstance(image, Image.Image):
                # Convert PIL Image to tensor (handles RGB correctly)
                image = transforms.functional.to_tensor(image)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to tensor and normalize to [0, 1]
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image

        return transform

    def _train_one_epoch(self, model, optimizer, data_loader):
        """
        Train model for one complete epoch.

        Args:
            model (torch.nn.Module): Model to train
            optimizer (torch.optim.Optimizer): Optimizer for parameter updates
            data_loader (DataLoader): Training data loader

        Returns:
            float: Average training loss for the epoch

        Performs one complete pass through the training data:
            1. Forward pass to compute losses
            2. Backward pass to compute gradients
            3. Optimizer step to update parameters
            4. Average loss across all batches
        """
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        running_loss = 0.0

        for images, targets in data_loader:
            # Move data to device (GPU/CPU)
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass - model returns loss dict in training mode
            # Dict contains classification loss, box regression loss, etc.
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            losses.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            running_loss += losses.item()

        return running_loss / len(data_loader)

    def _evaluate_model(self, model, data_loader):
        """
        Evaluate model on validation data.

        Args:
            model (torch.nn.Module): Model to evaluate
            data_loader (DataLoader): Validation data loader

        Returns:
            float: Average validation loss

        Computes validation loss without updating model parameters.
        Model is kept in training mode to compute losses properly
        (Faster R-CNN requires this).
        """
        model.train()  # Keep in train mode for loss calculation
        running_val_loss = 0.0

        # Enable gradients for loss computation (but don't update weights)
        with torch.set_grad_enabled(True):
            for images, targets in data_loader:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass to get losses
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_val_loss += losses.item()

        return running_val_loss / len(data_loader)

    def _calculate_time_estimates(self, start_time, k_folds, fold, num_epochs, epoch):
        """
        Calculate training progress and time estimates.

        Args:
            start_time (float): Training start timestamp (from time.time())
            k_folds (int): Total number of cross-validation folds
            fold (int): Current fold index (0-based)
            num_epochs (int): Number of epochs per fold
            epoch (int): Current epoch index (0-based)

        Returns:
            tuple: (elapsed_time, estimated_remaining, avg_time_per_epoch)
                - elapsed_time: Seconds since training started
                - estimated_remaining: Estimated seconds until completion
                - avg_time_per_epoch: Average seconds per epoch so far

        Provides useful progress information for long training runs.
        """
        # Calculate elapsed time
        current_time = time.time() - start_time

        # Calculate progress through total training
        total_epochs = k_folds * num_epochs
        current_epoch_global = fold * num_epochs + (epoch + 1)

        # Estimate remaining time based on average so far
        avg_time_per_epoch = current_time / current_epoch_global
        remaining_epochs = total_epochs - current_epoch_global
        estimated_remaining = remaining_epochs * avg_time_per_epoch

        return current_time, estimated_remaining, avg_time_per_epoch

    # ================================
    # INLINE FILTERING METHODS
    # ================================

    def _is_valid_detection(self, box, score, mask, img_width, img_height):
        """
        Check if a detection passes ALL filtering criteria in one pass.

        Args:
            box (array): Bounding box [xmin, ymin, xmax, ymax]
            score (float): Confidence score [0, 1]
            mask (np.ndarray): Binary mask (True = masked/invalid) or None
            img_width (int): Image width in pixels
            img_height (int): Image height in pixels

        Returns:
            bool: True if detection passes all filters, False otherwise

        Filters Applied (in order of computational cost):
            1. Confidence threshold
            2. Minimum size threshold
            3. Edge proximity check
            4. Mask overlap check
        """
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Filter 1: Confidence threshold (cheapest check first)
        if score < self.config.confidence_threshold:
            return False

        # Filter 2: Minimum size threshold
        # Reject very small detections that are likely noise
        if self.config.min_iceberg_size > 0:
            if width * height < self.config.min_iceberg_size:
                return False

        # Filter 3: Edge proximity check
        # Reject detections too close to image edges (often partial/cut-off objects)
        if self.config.edge_tolerance > 0:
            if (xmin <= self.config.edge_tolerance or
                    ymin <= self.config.edge_tolerance or
                    xmax >= img_width - self.config.edge_tolerance or
                    ymax >= img_height - self.config.edge_tolerance):
                return False

        # Filter 4: Mask overlap check (most expensive, do last)
        # Reject detections that overlap too much with masked regions (e.g., land)
        if mask is not None and self.config.filter_masked_regions:
            # Convert to integer coordinates and clamp to image bounds
            left = int(max(0, xmin))
            top = int(max(0, ymin))
            right = int(min(xmax, img_width))
            bottom = int(min(ymax, img_height))

            # Check for valid box dimensions
            if right <= left or bottom <= top:
                return False

            # Clamp to mask boundaries (mask might be different size)
            left = max(0, min(left, mask.shape[1] - 1))
            right = max(left + 1, min(right, mask.shape[1]))
            top = max(0, min(top, mask.shape[0] - 1))
            bottom = max(top + 1, min(bottom, mask.shape[0]))

            # Extract mask region for this bounding box
            submask = mask[top:bottom, left:right]

            if submask.size > 0:
                # Count pixels in masked region (True = masked/land)
                masked_pixel_count = np.count_nonzero(submask)
                # Calculate ratio of water pixels (not masked)
                water_ratio = 1 - masked_pixel_count / float(submask.size)

                # Reject if too much of detection is on land
                if water_ratio > self.config.mask_ratio_threshold:
                    return False

        return True

    # ================================
    # DETECTION UTILITIES
    # ================================

    def _calculate_iou(self, box1, box2):
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
                - 0.5+: Significant overlap (common NMS threshold)

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

    def _calculate_iou_batch(self, boxes1, boxes2):
        """
        Calculate IoU between two sets of bounding boxes efficiently.

        Vectorized implementation using numpy broadcasting for computing
        IoU between all pairs of boxes. Much faster than nested loops.

        Args:
            boxes1 (np.ndarray): First set of boxes [N, 4] in (x1, y1, x2, y2) format
            boxes2 (np.ndarray): Second set of boxes [M, 4] in (x1, y1, x2, y2) format

        Returns:
            np.ndarray: IoU matrix [N, M] where element (i, j) is IoU between
                       boxes1[i] and boxes2[j]
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([]).reshape(len(boxes1), len(boxes2))

        # Expand dimensions for broadcasting
        # boxes1: [N, 1, 4], boxes2: [1, M, 4]
        boxes1 = np.expand_dims(boxes1, axis=1)
        boxes2 = np.expand_dims(boxes2, axis=0)

        # Calculate intersection coordinates (broadcasts to [N, M, 1])
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

        # Calculate intersection and union areas
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection

        # Avoid division by zero
        return intersection / (union + 1e-6)

    def _calculate_intersection(self, box1, box2):
        """
        Calculate intersection area between two bounding boxes.

        Args:
            box1 (tuple): First bounding box (x1, y1, x2, y2)
            box2 (tuple): Second bounding box (x1, y1, x2, y2)

        Returns:
            float: Intersection area in pixels² (0 if no overlap)

        Used for nested detection removal where we need the intersection
        area directly rather than IoU ratio.
        """
        # Find intersection rectangle coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Check if boxes actually intersect
        if x2 <= x1 or y2 <= y1:
            return 0.0

        return (x2 - x1) * (y2 - y1)

    def _nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        NMS is a post-processing technique that removes redundant detections
        by keeping only the highest-scoring detection in each group of
        highly overlapping detections.

        Args:
            detections (list): List of detection dicts with 'box' and 'score' keys
            iou_threshold (float): IoU threshold for suppression
                Higher values = more permissive (keep more overlapping boxes)

        Returns:
            list: Filtered detections after NMS

        Algorithm:
            1. Sort detections by confidence (highest first)
            2. Keep highest scoring detection
            3. Remove all detections that overlap significantly (IoU > threshold)
            4. Repeat with remaining detections
        """
        if not detections:
            return []

        # Sort by confidence score (highest first)
        detections.sort(key=lambda x: x['score'], reverse=True)
        keep = []

        while detections:
            # Keep the highest-scoring detection
            best = detections.pop(0)
            keep.append(best)

            # Remove highly overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou(best['box'], det['box']) < iou_threshold
            ]

        return keep

    # ================================
    # INFERENCE METHODS WITH INLINE FILTERING
    # ================================

    def _run_multi_scale_prediction(self, img_path, confidence_threshold, mask=None,
                                    img_width=None, img_height=None):
        """
        Run multi-scale detection with inline filtering.

        Multi-scale detection improves robustness by testing the image at
        multiple scales. This helps detect objects that might be too small
        or too large at the original scale.

        Args:
            img_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for detections
            mask (np.ndarray, optional): Binary mask for filtering
            img_width (int, optional): Image width for filtering
            img_height (int, optional): Image height for filtering

        Returns:
            list: Valid detections in standardized format [x, y, width, height]

        Process:
            1. For each scale factor:
               a. Resize image
               b. Run detection
               c. Scale boxes back to original size
               d. Apply inline filtering
            2. Apply NMS to remove cross-scale duplicates
            3. Return filtered detections

        The inline filtering ensures only valid detections are kept
        before the expensive NMS operation.
        """
        # Load and prepare image
        original_img = Image.open(img_path).convert("RGB")
        original_size = original_img.size

        # Get image dimensions
        if img_width is None or img_height is None:
            img_width, img_height = original_size

        scale_detections = []

        # Test at each scale
        for scale in self.config.scales:
            # Resize image
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_img = original_img.resize(new_size, Image.LANCZOS)
            img_tensor = self._get_transforms()(scaled_img).unsqueeze(0).to(self.device)

            # Run detection
            with torch.no_grad():
                predictions = self.model(img_tensor)

            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            if len(boxes) > 0:
                # Scale boxes back to original image size
                boxes[:, [0, 2]] /= scale
                boxes[:, [1, 3]] /= scale

                # Apply ALL filters immediately - only keep valid detections
                for box, score in zip(boxes, scores):
                    if self._is_valid_detection(box, score, mask, img_width, img_height):
                        scale_detections.append({
                            'box': box,
                            'score': score,
                            'method': 'multi_scale'
                        })

        # NMS only on valid detections (much smaller set!)
        if scale_detections:
            merged = self._nms(scale_detections, iou_threshold=0.5)
            return self._convert_to_detection_format(merged)
        return []

    def _run_sliding_window_prediction(self, img_path, confidence_threshold, mask=None,
                                       img_width=None, img_height=None):
        """
        Run sliding window detection with inline filtering.

        Sliding window detection processes large images by breaking them into
        overlapping windows. This is essential for high-resolution imagery
        where full-image processing would be too memory-intensive.

        Args:
            img_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for detections
            mask (np.ndarray, optional): Binary mask for filtering
            img_width (int, optional): Image width for filtering
            img_height (int, optional): Image height for filtering

        Returns:
            list: Valid detections in standardized format [x, y, width, height]

        Process:
            1. Slide window across image with overlap
            2. For each window:
               a. Extract window region
               b. Run detection
               c. Convert to global coordinates
               d. Apply inline filtering
            3. Apply NMS to remove overlaps
            4. Return filtered detections

        The overlap between windows ensures objects near window boundaries
        are still detected. Inline filtering reduces the number of
        detections that need NMS processing.
        """
        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Get image dimensions
        if img_width is None or img_height is None:
            img_width, img_height = w, h

        window_detections = []
        window_size = self.config.window_size
        overlap = self.config.overlap

        # Calculate step sizes based on overlap
        step_x = int(window_size[0] * (1 - overlap))
        step_y = int(window_size[1] * (1 - overlap))

        # Slide window across image
        for y in range(0, h - window_size[1] + 1, step_y):
            for x in range(0, w - window_size[0] + 1, step_x):
                # Extract window
                window = img_rgb[y:y + window_size[1], x:x + window_size[0]]
                window_pil = Image.fromarray(window)
                window_tensor = self._get_transforms()(window_pil).unsqueeze(0).to(self.device)

                # Run detection on window
                with torch.no_grad():
                    predictions = self.model(window_tensor)

                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()

                # Convert to global coordinates and filter immediately
                for box, score in zip(boxes, scores):
                    # Convert from window coordinates to global image coordinates
                    global_box = np.array([box[0] + x, box[1] + y, box[2] + x, box[3] + y])

                    # Apply ALL filters immediately
                    if self._is_valid_detection(global_box, score, mask, img_width, img_height):
                        window_detections.append({
                            'box': global_box,
                            'score': score,
                            'method': 'sliding_window'
                        })

        # NMS only on valid detections
        if window_detections:
            merged = self._nms(window_detections, iou_threshold=0.5)
            return self._convert_to_detection_format(merged)
        return []

    def _remove_overlaps(self, multi_scale_dets, sliding_window_dets, iou_threshold):
        """
        Intelligently combine detections from both methods with priority rules.

        When combining multi-scale and sliding window detections, we need to
        handle overlaps intelligently. This method implements smart priority
        rules to keep the best detections from each method.

        Args:
            multi_scale_dets (list): Detections from multi-scale method
            sliding_window_dets (list): Detections from sliding window method
            iou_threshold (float): IoU threshold for considering boxes as overlapping

        Returns:
            list: Final combined and filtered detections

        Priority Rules:
            1. Multi-scale detections have priority (generally better quality)
            2. Among sliding window detections, larger boxes win
            3. Non-overlapping detections from both methods are kept

        This approach ensures we get comprehensive coverage while
        removing redundant detections intelligently.
        """
        # Start with all multi-scale detections (they have priority)
        final_detections = multi_scale_dets.copy()

        # Convert multi-scale detections to box array for efficient comparison
        ms_boxes = []
        if multi_scale_dets:
            ms_boxes = np.array([[d['x'], d['y'], d['x'] + d['width'], d['y'] + d['height']]
                                 for d in multi_scale_dets])

        # Process each sliding window detection
        for sw_det in sliding_window_dets:
            sw_box = np.array([sw_det['x'], sw_det['y'],
                               sw_det['x'] + sw_det['width'], sw_det['y'] + sw_det['height']])

            should_keep = True

            # Check overlap with multi-scale detections (multi-scale wins)
            if len(ms_boxes) > 0:
                ious = self._calculate_iou_batch(sw_box.reshape(1, -1), ms_boxes)[0]
                if np.any(ious > iou_threshold):
                    should_keep = False

            # Check overlap with already accepted sliding window detections
            if should_keep and len(final_detections) > len(multi_scale_dets):
                # Build list of already accepted sliding window boxes
                accepted_sw_boxes = []
                for det in final_detections[len(multi_scale_dets):]:
                    box = np.array([det['x'], det['y'], det['x'] + det['width'], det['y'] + det['height']])
                    accepted_sw_boxes.append(box)

                if accepted_sw_boxes:
                    accepted_sw_boxes = np.array(accepted_sw_boxes)
                    ious = self._calculate_iou_batch(sw_box.reshape(1, -1), accepted_sw_boxes)[0]
                    overlap_indices = np.where(ious > iou_threshold)[0]

                    if len(overlap_indices) > 0:
                        # Compare areas - larger bounding box wins
                        sw_area = sw_det['width'] * sw_det['height']

                        # Check if current detection is larger than all overlapping ones
                        larger_than_all = True
                        indices_to_remove = []

                        for ov_idx in overlap_indices:
                            existing_det = final_detections[len(multi_scale_dets) + ov_idx]
                            existing_area = existing_det['width'] * existing_det['height']

                            if sw_area <= existing_area:
                                # Current detection is not larger, don't keep it
                                larger_than_all = False
                                break
                            else:
                                # Current detection is larger, mark smaller one for removal
                                indices_to_remove.append(len(multi_scale_dets) + ov_idx)

                        if larger_than_all:
                            # Remove smaller overlapping detections
                            for idx in sorted(indices_to_remove, reverse=True):
                                final_detections.pop(idx)
                        else:
                            should_keep = False

            # Add detection if it passed all filters
            if should_keep:
                final_detections.append(sw_det)

        return final_detections

    def _remove_nested_detections(self, detection_path):
        """
        Remove detections that are fully enclosed within larger detections.

        Sometimes the model detects both a large iceberg and smaller fragments
        or pieces within it. This method removes nested detections to clean
        up the results, keeping only the larger parent detection.

        Special Case:
            If a smaller nested detection has significantly higher confidence
            (>0.5) than the larger box (<0.5), we keep the smaller box instead.
            This handles cases where the model is more confident about a
            fragment than the whole iceberg.

        Args:
            detection_path (str): Path to detection file to clean

        Returns:
            int: Number of detections removed

        Process:
            1. Load detections and group by frame
            2. For each frame:
               a. Sort detections by area (largest first)
               b. Check each detection against larger ones
               c. Remove if nested (with confidence exception)
            3. Save cleaned detections

        Threshold:
            A detection is considered nested if ≥95% of its area overlaps
            with a larger detection.
        """
        # Load detections grouped by frame
        detections_by_frame = load_icebergs_by_frame(detection_path)

        # Process each frame independently
        all_filtered_detections = []
        pre_detections_count = 0

        for frame_id, frame_detections in detections_by_frame.items():
            # Convert to comparable format and calculate areas
            pre_detections_count += len(frame_detections)
            boxes_data = []

            for object_id in frame_detections:
                det = frame_detections[object_id]
                x, y, width, height = det["bbox"]

                boxes_data.append({
                    'image': frame_id,
                    'object_id': object_id,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'x2': x + width,
                    'y2': y + height,
                    'score': det['conf'],
                    'area': width * height,
                    'keep': True  # Initialize as keep=True
                })

            # Sort by area (largest first) - ensures we process large boxes before small
            boxes_data.sort(key=lambda x: x['area'], reverse=True)

            # Track kept detections with their properties
            kept_boxes = []

            # Check each detection against already kept (larger) ones
            for current_idx, item in enumerate(boxes_data):
                # Skip if already marked as not keep
                if not item['keep']:
                    continue

                current_box = item["x"], item["y"], item["x2"], item["y2"]
                current_area = item['area']
                current_conf = item['score']
                is_nested = False

                # Only check if confidence is not very high
                if current_conf < 0.95:
                    # Check against all kept boxes (which are all larger)
                    for kept_idx, (kept_box, kept_area, kept_conf) in enumerate(kept_boxes):
                        # Calculate intersection
                        intersection = self._calculate_intersection(current_box, kept_box)

                        # Check if most of the current (smaller) box is inside the kept (larger) box
                        coverage = intersection / current_area if current_area > 0 else 0.0

                        if coverage >= self.config.max_iceberg_overlap:
                            # Found a nested detection
                            # Special case: smaller box has high confidence, larger box has low confidence
                            if kept_conf < 0.5 and current_conf > 0.5:
                                # Keep current (smaller) box, remove kept (larger) box
                                # Find the kept box in boxes_data and mark it as not keep
                                for box in boxes_data:
                                    box_tuple = (box['x'], box['y'], box['x2'], box['y2'])
                                    if box_tuple == kept_box and box['area'] == kept_area:
                                        box['keep'] = False
                                        break

                                # Remove from kept_boxes list
                                kept_boxes.pop(kept_idx)
                                # Don't mark current as nested
                                is_nested = False
                            else:
                                # Normal case: remove the smaller nested box
                                is_nested = True

                            break

                if not is_nested:
                    kept_boxes.append((current_box, item['area'], item['score']))
                else:
                    item['keep'] = False

            # Add kept detections from this frame
            frame_filtered = [item for item in boxes_data if item['keep']]
            all_filtered_detections.extend(frame_filtered)

        # Sort by original index to maintain input order
        all_filtered_detections.sort(key=lambda x: x['object_id'])

        # Log overall statistics
        total_removed = pre_detections_count - len(all_filtered_detections)
        if total_removed > 0:
            logger.info(f"\nTotal removed across all frames: {total_removed} nested detections "
                        f"({total_removed / pre_detections_count * 100:.1f}%)")

        # Save cleaned detections
        self._save_detections(all_filtered_detections, detection_path)
        return total_removed

    def _convert_to_detection_format(self, detections):
        """
        Convert detections to standardized output format.

        Args:
            detections (list): Raw detections with 'box' [x1,y1,x2,y2] and 'score' keys

        Returns:
            list: Detections in standardized format with x, y, width, height

        Converts from corner coordinate format [xmin, ymin, xmax, ymax]
        to position+size format [x, y, width, height] used throughout
        the system and in output files.
        """
        formatted = []
        for det in detections:
            box = det['box']
            xmin, ymin, xmax, ymax = box
            formatted.append({
                'x': xmin,
                'y': ymin,
                'width': xmax - xmin,
                'height': ymax - ymin,
                'score': det['score'],
                'method': det.get('method', 'unknown')
            })
        return formatted

    def _save_detections(self, detections, detections_file):
        """
        Save detections to file in MOTChallenge format.

        Args:
            detections (list): List of detection dictionaries
            detections_file (str): Path to output file

        Output Format:
            Each line: frame,object_id,x,y,width,height,confidence,1,-1,-1
            Where:
                - frame: Frame/image identifier
                - object_id: Detection ID (unique within frame)
                - x,y,width,height: Bounding box
                - confidence: Detection confidence score
                - 1: Class ID (all icebergs)
                - -1,-1: Unused fields for MOTChallenge compatibility

        The file is sorted by frame ID after writing for consistency.
        """
        with open(detections_file, 'w') as f:
            for det in detections:
                f.write(f"{det['image']},{det['object_id']},{det['x']},{det['y']},"
                        f"{det['width']},{det['height']},{det['score']},1,-1,-1\n")

        # Sort file by frame ID for consistency with evaluation tools
        sort_file(detections_file)