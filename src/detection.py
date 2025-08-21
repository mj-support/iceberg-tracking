import copy
import cv2
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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Tuple, List

from embedding import IcebergEmbeddingsConfig, IcebergEmbeddingsTrainer
from utils.helpers import DATA_DIR, PROJECT_ROOT, sort_file

"""
Iceberg Detection System using Faster R-CNN

This module provides a comprehensive system for detecting icebergs in timelapse images
using a Faster R-CNN deep learning model. The system includes training with cross-validation,
multi-scale inference, sliding window detection, and postprocessing capabilities.

Key Features:
- K-fold cross-validation training
- Multi-scale detection for robustness
- Sliding window inference for large images  
- Intelligent overlap removal between detection methods
- Optional masking and postprocessing
- Embedding generation for detected icebergs
"""


# ================================
# CONFIGURATION CLASS
# ================================

@dataclass
class IcebergDetectionConfig:
    """
    Configuration class to centralize all hyperparameters and settings for iceberg detection.

    This class uses dataclass to provide a clean interface for configuring all aspects
    of the detection pipeline, from data loading to model training and inference.

    Attributes:
        dataset (str): Name of the dataset directory
        image_format (str): File extension for images (e.g., 'JPG', 'PNG')
        masking (bool): Whether to apply masking during preprocessing to remove landareas
        num_workers (int): Number of worker processes for data loading

        # Model Architecture Parameters
        num_classes (int): Number of classes (background + iceberg = 2)
        anchor_sizes (Tuple): Anchor box sizes for different feature pyramid levels
        anchor_aspect_ratios (Tuple): Aspect ratios for anchor boxes

        # Training Parameters
        k_folds (int): Number of folds for cross-validation
        num_epochs (int): Maximum epochs per fold
        patience (int): Early stopping patience (epochs without improvement)
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate for optimizer
        momentum (float): SGD momentum parameter
        weight_decay (float): L2 regularization weight

        # Detection Parameters
        box_detections_per_img (int): Maximum detections per image
        box_nms_thresh (float): NMS threshold for final detections
        rpn_nms_thresh (float): NMS threshold for RPN proposals
        rpn_score_thresh (float): Score threshold for RPN proposals

        # Inference Parameters
        confidence_threshold (float): Minimum confidence for keeping detections
        scales (List[float]): Scales for multi-scale detection
        window_size (Tuple[int, int]): Size of sliding windows
        overlap (float): Overlap ratio between sliding windows
        iou_threshold (float): IoU threshold for removing duplicate detections

        # Postprocessing Parameters
        postprocess (bool): Whether to apply postprocessing filters
        edge_tolerance (int): Pixel tolerance for edge detection filtering
        mask_ratio_threshold (float): Threshold for mask-based filtering
        feature_extraction (bool): Whether to extract embeddings from detections

        # Hardware Configuration
        device (str): Computing device (GPU/CPU)
    """
    # Data configuration
    dataset: str
    image_format: str = "JPG"
    masking: bool = False
    num_workers: int = 4

    # Model parameters - These define the Faster R-CNN architecture
    num_classes: int = 2  # Background + iceberg
    anchor_sizes: Tuple[Tuple[int, ...], ...] = ((16,), (32,), (64,), (128,), (256,))
    anchor_aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 5

    # Training parameters - Control the training process
    k_folds: int = 5
    num_epochs: int = 10
    patience: int = 3  # Early stopping patience
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
    confidence_threshold: float = 0.0
    scales: List[float] = None  # Will be set in __post_init__
    window_size: Tuple[int, int] = (1024, 1024)
    overlap: float = 0.2
    iou_threshold: float = 0.3

    # Postprocessing parameters - Control filtering of detections
    postprocess: bool = True
    edge_tolerance: int = 0
    mask_ratio_threshold: float = 0.02
    feature_extraction: bool = True

    # Device configuration
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __post_init__(self):
        """
        Set default values for mutable types after initialization.

        This is necessary because mutable default arguments in dataclasses
        can cause issues. We set them here after object creation.
        """
        if self.scales is None:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]


# ================================
# DATASET CLASS
# ================================
class IcebergDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for iceberg detection with improved error handling and logging.

    This dataset class handles both training mode (with annotations) and inference mode
    (image-only). It provides robust error handling and can recover from corrupted
    images by skipping to the next available image.

    Args:
        image_dir (str): Directory containing input images
        det_file (str, optional): Path to detection/annotation file for training
        transforms (callable, optional): Transform function to apply to images
        config (IcebergDetectionConfig): Configuration object

    Attributes:
        img_folder (str): Path to image directory
        image_format (str): File extension for images
        transforms (callable): Image transformation function
        config (IcebergDetectionConfig): Configuration settings
        detections (pd.DataFrame): Loaded annotations (training mode only)
        unique_images (List[str]): List of unique image names
    """

    def __init__(self, image_dir, det_file=None, transforms=None, config=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Directory containing input images
            det_file (str, optional): Path to annotations file (for training)
            transforms (callable, optional): Image transformations to apply
            config (IcebergDetectionConfig): Configuration object
        """
        self.img_folder = image_dir
        self.image_format = f".{config.image_format}"
        self.transforms = transforms
        self.config = config
        self.detections = None
        self.unique_images = []

        # Load data based on whether we have annotations (training vs inference)
        self._load_data(det_file)

    def _load_data(self, det_file):
        """
        Load annotations or image list based on mode.

        Args:
            det_file (str, optional): Path to detection file with annotations

        In training mode (det_file exists), loads annotations from CSV file.
        In inference mode (no det_file), loads list of image files for processing.
        """
        if det_file and os.path.exists(det_file):
            # Training mode - load annotations
            column_names = ['image', 'iceberg_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                            'conf', 'unused_1', 'unused_2', 'unused_3']
            self.detections = pd.read_csv(det_file, names=column_names)
            self.unique_images = self.detections['image'].unique().tolist()
            print(f"Loaded {len(self.unique_images)} images with annotations")
        else:
            # Inference mode - load image file list
            self.unique_images = [
                os.path.splitext(f)[0] for f in os.listdir(self.img_folder)
                if f.endswith(self.image_format)
            ]
            print(f"Loaded {len(self.unique_images)} images for inference")

    def __getitem__(self, index):
        """
        Get dataset item with comprehensive error handling.

        Args:
            index (int): Index of item to retrieve

        Returns:
            tuple: (image, target) for training mode or (image, metadata) for inference

        The method implements robust error recovery - if an image fails to load,
        it automatically tries the next image in the sequence to avoid crashes.
        """
        try:
            img_name = self.unique_images[index]
            img_file = os.path.join(self.img_folder, f"{img_name}{self.image_format}")

            # Check if image file exists
            if not os.path.exists(img_file):
                raise FileNotFoundError(f"Image file not found: {img_file}")

            # Load and convert image to RGB
            img = Image.open(img_file).convert("RGB")
            image_id = torch.tensor([index])

            # Inference mode - return image and metadata only
            if self.detections is None:
                if self.transforms:
                    img = self.transforms(img)
                return img, {"image_id": image_id, "file_name": img_name}

            # Training mode - process annotations
            return self._process_training_item(img, img_name, image_id, index)

        except Exception as e:
            print(f"Error loading item {index}: {e}")
            # Fallback: try next item to avoid training crashes
            return self.__getitem__((index + 1) % len(self.unique_images))

    def _process_training_item(self, img, img_name, image_id, index):
        """
        Process training item with annotations.

        Args:
            img (PIL.Image): Loaded image
            img_name (str): Name of the image file
            image_id (torch.Tensor): Unique image identifier
            index (int): Current index (for fallback if no valid boxes)

        Returns:
            tuple: (image, target_dict) where target_dict contains bounding boxes,
                   labels, and other metadata required by Faster R-CNN
        """
        # Get all detections for this image
        img_detections = self.detections[self.detections['image'] == img_name]
        boxes, labels = self._extract_valid_boxes(img_detections)

        # Skip images without valid annotations
        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self.unique_images))

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Calculate areas for each bounding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Create iscrowd tensor (all zeros - no crowd annotations)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # Create target dictionary as expected by Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'file_name': img_name
        }

        # Apply transformations if specified
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def _extract_valid_boxes(self, img_detections):
        """
        Extract valid bounding boxes from annotations.

        Args:
            img_detections (pd.DataFrame): Detections for a single image

        Returns:
            tuple: (boxes, labels) where boxes is list of [xmin, ymin, xmax, ymax]
                   and labels is list of class labels (all 1 for iceberg)

        Filters out invalid boxes (zero or negative width/height) that could
        cause training instability.
        """
        boxes = []
        labels = []

        for _, det in img_detections.iterrows():
            xmin, ymin = det['bb_left'], det['bb_top']
            width, height = det['bb_width'], det['bb_height']

            # Only keep boxes with positive dimensions
            if width > 0 and height > 0:
                boxes.append([xmin, ymin, xmin + width, ymin + height])
                labels.append(1)  # Iceberg class (background is 0)

        return boxes, labels

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.unique_images)


# ================================
# MAIN DETECTOR CLASS
# ================================
class IcebergDetector:
    """
    Main iceberg detection class with integrated training, inference, and postprocessing.

    This class provides a complete pipeline for iceberg detection:
    1. Training with k-fold cross-validation
    2. Multi-scale and sliding window inference
    3. Intelligent overlap removal between detection methods
    4. Postprocessing with masking and edge filtering
    5. Feature extraction from detected icebergs

    The system is designed to be robust and production-ready, with comprehensive
    error handling, progress tracking, and configurable parameters.

    Args:
        config (IcebergDetectionConfig): Configuration object with all parameters

    Attributes:
        config (IcebergDetectionConfig): Configuration settings
        dataset (str): Dataset name
        image_format (str): Image file extension
        masking (bool): Whether masking is enabled
        device (torch.device): Computing device (GPU/CPU)
        model (torch.nn.Module): The trained Faster R-CNN model

        # File paths
        model_file (str): Path to saved model weights
        image_dir (str): Directory containing input images
        annotations_file (str): Path to ground truth annotations
        detections_file (str): Path to output detections
        embeddings_path (str): Path to extracted feature embeddings
    """

    def __init__(self, config: IcebergDetectionConfig):
        """
        Initialize the iceberg detector with configuration.

        Sets up all necessary directories and file paths based on the dataset
        configuration. Creates output directories if they don't exist.

        Args:
            config (IcebergDetectionConfig): Configuration object with all parameters
        """
        self.config = config
        self.dataset = config.dataset
        self.image_format = f".{config.image_format}"
        self.masking = config.masking
        self.device = config.device
        self.model = None

        # Create necessary output directories
        base_path = os.path.join(DATA_DIR, self.dataset)
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "detections"), exist_ok=True)

        # Define all file paths used throughout the pipeline
        self.model_file = os.path.join(PROJECT_ROOT, "models", "iceberg_detector.pth")
        self.image_dir = os.path.join(base_path, "images", "raw")
        self.annotations_file = os.path.join(base_path, "annotations", "gt.txt")
        self.detections_file = os.path.join(base_path, "detections", "det.txt")
        self.embeddings_path = os.path.join(DATA_DIR, self.dataset, "embeddings", "det_embeddings.pt")
        self.iceberg_embedding_model = os.path.join(PROJECT_ROOT, "models", "embedding_model.pth")

        # Set up mask file path if masking is enabled
        if self.masking:
            self.mask_file = os.path.join(DATA_DIR, self.dataset, "images", f"mask{self.image_format}")

        print(f"Initialized IcebergDetector for dataset: {self.dataset}")
        print(f"Device: {self.device}")

    def preprocessing(self):
        """
        Apply preprocessing (masking) if enabled in configuration.

        This method applies image masking to remove unwanted regions (like land areas)
        from satellite imagery before training or inference. The masked images are
        saved to a separate 'processed' directory to avoid modifying originals.

        The masking process:
        1. Loads a binary mask image
        2. For each input image, sets masked pixels to black (0,0,0)
        3. Saves processed images to output directory
        4. Updates image_dir to point to processed images
        """
        if not self.masking:
            return

        output_dir = os.path.join(DATA_DIR, self.dataset, "images", "processed")
        os.makedirs(output_dir, exist_ok=True)

        # Only process if output directory is empty
        if not any(os.scandir(output_dir)):
            print("\nStarting preprocessing: applying masks")
            self._apply_masking(output_dir)
            self.image_dir = output_dir
            print("\nFinished preprocessing")

    def train(self):
        """
        Train the model with k-fold cross-validation.

        This method implements a comprehensive training pipeline:
        1. Preprocessing (masking if enabled)
        2. Dataset creation with proper transforms
        3. K-fold cross-validation to ensure robust model selection
        4. Early stopping to prevent overfitting
        5. Progress tracking with time estimates
        6. Model selection based on validation loss

        The training uses SGD optimizer with momentum and tracks both training
        and validation losses. The best model across all folds is saved.
        """
        # Extract configuration parameters
        k_folds = self.config.k_folds
        num_epochs = self.config.num_epochs
        patience = self.config.patience
        batch_size = self.config.batch_size
        learning_rate = self.config.learning_rate

        print("=== TRAINING PHASE ===")
        print(f"\nStarting training with {k_folds}-fold cross-validation")

        # Apply preprocessing (masking) if configured
        self.preprocessing()

        # Create training dataset with transforms
        dataset = IcebergDetectionDataset(
            image_dir=self.image_dir,
            det_file=self.annotations_file,
            transforms=self._get_transforms(),
            config=self.config
        )

        # Initialize cross-validation and tracking variables
        total_start_time = time.time()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_val_loss_overall = float('inf')
        best_model_state_overall = None

        # Cross-validation training loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"\nFold {fold + 1}/{k_folds}")

            # Create data subsets and loaders for this fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=4, collate_fn=lambda x: tuple(zip(*x))
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=4, collate_fn=lambda x: tuple(zip(*x))
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
            epochs_no_improve = 0

            # Epoch training loop
            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                # Train one epoch and evaluate
                train_loss = self._train_one_epoch(model, optimizer, train_loader)
                val_loss = self._evaluate_model(model, val_loader)

                # Calculate progress and time estimates
                current_time, estimated_remaining, avg_time = self._calculate_time_estimates(
                    total_start_time, k_folds, fold, num_epochs, epoch
                )

                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                      f"Time: {timedelta(seconds=int(current_time))}<"
                      f"{timedelta(seconds=int(estimated_remaining))}, "
                      f"{timedelta(seconds=int(avg_time))}/Epoch")

                # Check for improvement and update best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Early stopping if no improvement
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                    break

            # Update overall best model across all folds
            if best_val_loss < best_val_loss_overall:
                best_val_loss_overall = best_val_loss
                best_model_state_overall = best_model_state
                print(f"New best model with validation loss: {best_val_loss_overall:.4f}")

        # Save the best model found across all folds
        if best_model_state_overall:
            torch.save(best_model_state_overall, self.model_file)
            print(f"\nBest model saved with validation loss: {best_val_loss_overall:.4f}")
            print(f"Training completed in {timedelta(seconds=int(time.time() - total_start_time))}")

    def predict(self):
        """
        Run combined prediction approach using multiple detection strategies.

        This method implements a sophisticated inference pipeline that combines:
        1. Multi-scale detection: Runs inference at different image scales
        2. Sliding window detection: Processes large images in overlapping windows
        3. Intelligent overlap removal: Combines results with priority rules

        The combination approach provides better detection coverage and accuracy
        compared to using either method alone. Multi-scale detection is better
        for objects at different scales, while sliding window handles very large
        images and provides more fine-grained detection.
        """
        confidence_threshold = self.config.confidence_threshold

        print("\n=== INFERENCE PHASE ===")
        print("Starting combined prediction with intelligent overlap removal...")

        # Load the trained model
        self._load_model()
        self.model.eval()

        # Get list of images to process
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(self.image_format)]
        all_detections = []
        total_start_time = time.time()

        # Process each image
        for img_file in image_files:
            img_path = os.path.join(self.image_dir, img_file)
            img_name = os.path.splitext(img_file)[0]

            #print(f"Processing {img_name}...")

            # Run both detection methods
            multi_scale_dets = self._run_multi_scale_prediction(img_path, confidence_threshold)
            sliding_window_dets = self._run_sliding_window_prediction(img_path, confidence_threshold)

            #print(f"  Multi-scale: {len(multi_scale_dets)}, Sliding window: {len(sliding_window_dets)}")

            # Intelligently combine and remove overlapping detections
            final_detections = self._remove_overlaps(
                multi_scale_dets, sliding_window_dets, self.config.iou_threshold
            )

            #print(f"  Final: {len(final_detections)} detections")

            # Add metadata to each detection
            for i, det in enumerate(final_detections):
                det['image'] = img_name
                det['object_id'] = i + 1
                all_detections.append(det)

        # Save all detections to file
        self._save_detections(all_detections)
        print(f"\nDetected {len(all_detections)} detections across {len(image_files)} images")
        print(f"Inference completed in {timedelta(seconds=int(time.time() - total_start_time))}")

        # Run postprocessing if enabled
        if self.config.postprocess:
            self.postprocess()

        print(f"Combined detections saved to {self.detections_file}")
        print("\n=== COMPLETED ===")

    def postprocess(self):
        """
        Perform postprocessing on detections to filter false positives.

        This method applies several filtering strategies:
        1. Edge filtering: Remove detections too close to image edges
        2. Mask filtering: Remove detections in masked (invalid) regions
        3. Size filtering: Remove detections that are too small/large
        4. Feature extraction: Generate embeddings for remaining detections

        The postprocessing significantly reduces false positives while preserving
        valid iceberg detections, improving overall system precision.
        """
        print("\n=== POSTPROCESSING ===")
        if not self.config.postprocess:
            print("No postprocessing")
            return

        # Load mask and get image dimensions
        if self.masking:
            if not os.path.exists(self.mask_file):
                raise FileNotFoundError(f"Mask file not found: {self.mask_file}")
            else:
                # Load binary mask
                mask = cv2.imread(self.mask_file, cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(bool)
                image_height, image_width = mask.shape
        else:
            mask = None
            # Get image dimensions from a sample image
            random_image_file = [f for f in os.listdir(self.image_dir) if f.endswith(self.image_format)][0]
            random_image_file = os.path.join(self.image_dir, random_image_file)
            image_height, image_width = cv2.imread(random_image_file, cv2.IMREAD_GRAYSCALE).shape

        # Filter detections based on various criteria
        filtered_detections = []
        total_start_time = time.time()

        with open(self.detections_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    frame, id_, left, top, width, height = parts[:6]
                    conf = float (parts[6] if len(parts) > 6 else "1.0")

                    left, top, width, height = float(left), float(top), float(width), float(height)

                    # Apply filtering conditions
                    if (not self._touches_edge(left, top, width, height, image_width, image_height) and
                            not self._box_touches_mask(left, top, width, height, mask)) and conf >= self.config.confidence_threshold:
                        filtered_detections.append(line)

        # Write filtered detections back to file
        with open(self.detections_file, "w") as f:
            for line in filtered_detections:
                f.write(line)

        # Sort detections file for consistent output
        sort_file(self.detections_file)

        print(f"\nReduced detection across all images to {len(filtered_detections)} detections")

        # Extract features from filtered detections if enabled
        if self.config.feature_extraction:
            config = IcebergEmbeddingsConfig(dataset=self.dataset, image_format=self.image_format)
            trainer = IcebergEmbeddingsTrainer(config)
            trainer.generate_iceberg_embeddings(self.detections_file, self.embeddings_path)

        print(f"Postprocessing completed in {timedelta(seconds=int(time.time() - total_start_time))}")

    def _apply_masking(self, output_dir):
        """
        Apply masking to images by setting masked pixels to black.

        Args:
            output_dir (str): Directory to save masked images

        This method loads a binary mask and applies it to all images in the dataset.
        Masked regions (typically land areas in satellite imagery) are set to black
        to focus detection on valid regions (water areas where icebergs can exist).
        """
        if not os.path.exists(self.mask_file):
            raise FileNotFoundError(f"Mask file not found: {self.mask_file}")

        # Load mask image and convert to binary array
        mask_img = Image.open(self.mask_file).convert("RGB")
        mask_array = np.array(mask_img)
        # Create binary mask where black pixels are True
        mask = np.all(mask_array == [0, 0, 0], axis=-1)

        # Process all images in the dataset
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(self.image_format)]

        for img_name in image_files:
            img_path = os.path.join(self.image_dir, img_name)
            img = Image.open(img_path)
            img_array = np.array(img)

            # Apply mask by setting masked pixels to black
            masked_img = img_array.copy()
            masked_img[mask] = [0, 0, 0]

            # Save masked image
            masked_img = Image.fromarray(masked_img)
            masked_img.save(os.path.join(output_dir, img_name))

    # ================================
    # MODEL BUILDING METHODS
    # ================================

    def _build_model(self):
        """
        Build enhanced Faster R-CNN model with custom configuration.

        Creates a Faster R-CNN model with ResNet-50 FPN backbone, customized
        anchor generation, and proper class head for iceberg detection.

        Returns:
            torch.nn.Module: Configured Faster R-CNN model ready for training

        The model architecture includes:
        - ResNet-50 Feature Pyramid Network (FPN) backbone
        - Custom anchor generator with multiple scales and aspect ratios
        - Region Proposal Network (RPN) for object proposals
        - ROI head with classification and bounding box regression
        """
        # Create custom anchor generator with specified sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=self.config.anchor_sizes,
            aspect_ratios=self.config.anchor_aspect_ratios
        )

        # Build Faster R-CNN model with custom parameters
        model = fasterrcnn_resnet50_fpn(
            weights=None,  # Start without pretrained weights (loaded separately)
            rpn_anchor_generator=anchor_generator,
            box_detections_per_img=self.config.box_detections_per_img,
            box_nms_thresh=self.config.box_nms_thresh,
            rpn_nms_thresh=self.config.rpn_nms_thresh,
            rpn_score_thresh=self.config.rpn_score_thresh,
        )

        # Load compatible pretrained weights for faster convergence
        self._load_pretrained_weights(model)

        # Replace classifier head for our specific number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

        return model.to(self.device)

    def _load_pretrained_weights(self, model):
        """
        Load compatible pretrained weights while skipping incompatible layers.

        Args:
            model (torch.nn.Module): Model to load weights into

        This method carefully loads pretrained ImageNet weights while avoiding
        shape mismatches that can occur with custom anchor configurations or
        different number of classes. Only compatible layers are loaded.
        """
        pretrained_dict = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict()
        model_dict = model.state_dict()

        # Filter out keys with shape mismatches
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        # Update model dictionary and load filtered weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def _load_model(self):
        """
        Load trained model from saved weights file.

        Returns:
            torch.nn.Module: Loaded model in evaluation mode

        Raises:
            FileNotFoundError: If no trained model file exists

        This method loads a previously trained model for inference. It builds
        the model architecture and loads the saved state dict.
        """
        if self.model is None:
            self.model = self._build_model()

        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {self.model_file}")
        else:
            raise FileNotFoundError(f"No trained model found at {self.model_file}")

        return self.model

    # ================================
    # TRAINING METHODS
    # ================================

    def _get_transforms(self):
        """
        Get image transforms for training/inference.

        Returns:
            callable: Transform function that converts images to tensors

        Creates a transform function that handles both PIL Images and numpy arrays,
        converting them to PyTorch tensors with proper normalization (0-1 range).
        """

        def transform(image):
            if isinstance(image, Image.Image):
                # Convert PIL Image to tensor
                image = transforms.functional.to_tensor(image)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to tensor and normalize
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image

        return transform

    def _train_one_epoch(self, model, optimizer, data_loader):
        """
        Train model for one epoch.

        Args:
            model (torch.nn.Module): Model to train
            optimizer (torch.optim.Optimizer): Optimizer for parameter updates
            data_loader (DataLoader): Training data loader

        Returns:
            float: Average training loss for the epoch

        Performs one complete pass through the training data, computing losses
        and updating model parameters via backpropagation.
        """
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, targets in data_loader:
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items() if k != 'file_name'} for t in targets]

            # Forward pass - model returns loss dict in training mode
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

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
        Model is kept in training mode to compute losses properly.
        """
        model.train()  # Keep in train mode for loss calculation
        running_val_loss = 0.0

        with torch.set_grad_enabled(True):  # Enable gradients for loss computation
            for images, targets in data_loader:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items() if k != 'file_name'} for t in targets]

                # Forward pass to get losses
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_val_loss += losses.item()

        return running_val_loss / len(data_loader)

    def _calculate_time_estimates(self, start_time, k_folds, fold, num_epochs, epoch):
        """
        Calculate training progress and time estimates.

        Args:
            start_time (float): Training start timestamp
            k_folds (int): Total number of folds
            fold (int): Current fold index
            num_epochs (int): Epochs per fold
            epoch (int): Current epoch index

        Returns:
            tuple: (current_time, estimated_remaining, avg_time_per_epoch)

        Provides useful progress information including elapsed time,
        estimated remaining time, and average time per epoch.
        """
        current_time = time.time() - start_time
        total_epochs = k_folds * num_epochs
        current_epoch_global = fold * num_epochs + (epoch + 1)

        avg_time_per_epoch = current_time / current_epoch_global
        remaining_epochs = total_epochs - current_epoch_global
        estimated_remaining = remaining_epochs * avg_time_per_epoch

        return current_time, estimated_remaining, avg_time_per_epoch

    # ================================
    # DETECTION UTILITIES
    # ================================

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list): First bounding box [xmin, ymin, xmax, ymax]
            box2 (list): Second bounding box [xmin, ymin, xmax, ymax]

        Returns:
            float: IoU value between 0 and 1

        IoU is a standard metric for measuring bounding box overlap.
        Higher values indicate more overlap between boxes.
        """
        # Find intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Check if boxes actually intersect
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate intersection and union areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_iou_batch(self, boxes1, boxes2):
        """
        Calculate IoU between sets of bounding boxes efficiently.

        Args:
            boxes1 (np.ndarray): First set of boxes [N, 4]
            boxes2 (np.ndarray): Second set of boxes [M, 4]

        Returns:
            np.ndarray: IoU matrix [N, M]

        Vectorized implementation for computing IoU between all pairs
        of boxes in two sets. Much faster than nested loops.
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([]).reshape(len(boxes1), len(boxes2))

        # Expand dimensions for broadcasting
        boxes1 = np.expand_dims(boxes1, axis=1)
        boxes2 = np.expand_dims(boxes2, axis=0)

        # Calculate intersection coordinates
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

        # Calculate intersection and union areas
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _calculate_containment_batch(self, boxes1, boxes2):
        """
        Calculate containment ratio between sets of bounding boxes.

        Args:
            boxes1 (np.ndarray): First set of boxes [N, 4]
            boxes2 (np.ndarray): Second set of boxes [M, 4]

        Returns:
            np.ndarray: Containment matrix [N, M]

        Containment measures how much of boxes1 is contained within boxes2.
        Useful for detecting when smaller boxes are inside larger ones.
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([]).reshape(len(boxes1), len(boxes2))

        # Expand dimensions for broadcasting
        boxes1 = np.expand_dims(boxes1, axis=1)
        boxes2 = np.expand_dims(boxes2, axis=0)

        # Calculate intersection coordinates
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

        # Calculate containment ratio
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])

        return intersection / (area1 + 1e-6)

    def _nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            detections (list): List of detection dictionaries with 'box' and 'score'
            iou_threshold (float): IoU threshold for suppression

        Returns:
            list: Filtered detections after NMS

        NMS removes redundant detections by keeping only the highest-scoring
        detection in each group of highly overlapping detections.
        """
        if not detections:
            return []

        # Sort detections by confidence score (highest first)
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
    # INFERENCE METHODS
    # ================================

    def _run_multi_scale_prediction(self, img_path, confidence_threshold):
        """
        Run multi-scale detection on a single image.

        Args:
            img_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for detections

        Returns:
            list: Detections in standardized format

        Multi-scale detection helps find objects at different sizes by running
        inference on the same image at multiple scales. This is particularly
        useful for icebergs which can vary greatly in size.
        """
        original_img = Image.open(img_path).convert("RGB")
        original_size = original_img.size
        scale_detections = []

        # Run detection at each configured scale
        for scale in self.config.scales:
            # Resize image to current scale
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_img = original_img.resize(new_size, Image.LANCZOS)
            img_tensor = self._get_transforms()(scaled_img).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)

            # Extract predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            if len(boxes) > 0:
                # Scale boxes back to original image size
                boxes[:, [0, 2]] /= scale  # x coordinates
                boxes[:, [1, 3]] /= scale  # y coordinates

                # Filter by confidence and add to detections
                for box, score in zip(boxes, scores):
                    if score > confidence_threshold:
                        scale_detections.append({
                            'box': box,
                            'score': score,
                            'method': 'multi_scale'
                        })

        # Apply NMS to remove duplicates across scales
        if scale_detections:
            merged = self._nms(scale_detections, iou_threshold=0.5)
            return self._convert_to_detection_format(merged)
        return []

    def _run_sliding_window_prediction(self, img_path, confidence_threshold):
        """
        Run sliding window detection on a single image.

        Args:
            img_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for detections

        Returns:
            list: Detections in standardized format

        Sliding window detection processes large images by breaking them into
        overlapping windows. This approach can find small objects that might
        be missed when the entire image is downscaled.
        """
        # Load image using OpenCV for efficient processing
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        window_detections = []
        window_size = self.config.window_size
        overlap = self.config.overlap

        # Calculate step sizes based on overlap
        step_x = int(window_size[0] * (1 - overlap))
        step_y = int(window_size[1] * (1 - overlap))

        # Slide window across the image
        for y in range(0, h - window_size[1] + 1, step_y):
            for x in range(0, w - window_size[0] + 1, step_x):
                # Extract window
                window = img_rgb[y:y + window_size[1], x:x + window_size[0]]
                window_pil = Image.fromarray(window)
                window_tensor = self._get_transforms()(window_pil).unsqueeze(0).to(self.device)

                # Run inference on window
                with torch.no_grad():
                    predictions = self.model(window_tensor)

                # Process predictions
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()

                # Convert window-relative coordinates to global coordinates
                for box, score in zip(boxes, scores):
                    if score > confidence_threshold:
                        global_box = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
                        window_detections.append({
                            'box': global_box,
                            'score': score,
                            'method': 'sliding_window'
                        })

        # Apply NMS to remove duplicates from overlapping windows
        if window_detections:
            merged = self._nms(window_detections, iou_threshold=0.5)
            return self._convert_to_detection_format(merged)
        return []

    def _remove_overlaps(self, multi_scale_dets, sliding_window_dets, iou_threshold):
        """
        Intelligently remove overlapping detections with priority rules.

        Args:
            multi_scale_dets (list): Detections from multi-scale method
            sliding_window_dets (list): Detections from sliding window method
            iou_threshold (float): IoU threshold for overlap detection

        Returns:
            list: Final filtered detections

        This method combines detections from both approaches using intelligent
        priority rules:
        1. Multi-scale detections have priority over sliding window
        2. Among sliding window detections, larger boxes win over smaller ones
        3. Non-overlapping detections from both methods are kept
        """
        # Start with all multi-scale detections (they have priority)
        final_detections = multi_scale_dets.copy()

        # Convert multi-scale detections to box format for comparison
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
                                larger_than_all = False
                                break
                            else:
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

    def _convert_to_detection_format(self, detections):
        """
        Convert detections to standardized format.

        Args:
            detections (list): Raw detections with 'box' and 'score' keys

        Returns:
            list: Detections in standardized format with x, y, width, height

        Converts from [xmin, ymin, xmax, ymax] box format to [x, y, width, height]
        format used throughout the system.
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

    def _save_detections(self, detections):
        """
        Save detections in the required CSV format.

        Args:
            detections (list): List of detection dictionaries

        Saves detections to file in the standard format:
        image_name,object_id,x,y,width,height,confidence,-1,-1,-1
        """
        with open(self.detections_file, 'w') as f:
            for det in detections:
                f.write(f"{det['image']},{det['object_id']},{det['x']},{det['y']},"
                        f"{det['width']},{det['height']},{det['score']},-1,-1,-1\n")

    # ================================
    # POSTPROCESSING METHODS
    # ================================

    def _touches_edge(self, left, top, width, height, image_width, image_height):
        """
        Check if bounding box touches image edge within tolerance.

        Args:
            left, top, width, height (float): Bounding box parameters
            image_width, image_height (int): Image dimensions

        Returns:
            bool: True if box touches edge, False otherwise

        Edge detections are often false positives caused by partial objects
        at image boundaries. This filter removes such detections.
        """
        right = left + width
        bottom = top + height

        return (left <= self.config.edge_tolerance or
                top <= self.config.edge_tolerance or
                right >= image_width - self.config.edge_tolerance or
                bottom >= image_height - self.config.edge_tolerance)

    def _box_touches_mask(self, left, top, width, height, mask):
        """
        Check if bounding box significantly overlaps with masked area.

        Args:
            left, top, width, height (float): Bounding box parameters
            mask (np.ndarray): Binary mask array (True = masked)

        Returns:
            bool: True if box overlaps significantly with mask

        Uses mask ratio threshold to determine if a detection overlaps too much
        with invalid/masked regions (like land areas in satellite imagery).
        """
        if mask is None:
            return False

        # Convert to integer coordinates and ensure valid bounds
        left, top = int(round(left)), int(round(top))
        right, bottom = int(round(left + width)), int(round(top + height))

        # Clamp to image boundaries
        h, w = mask.shape
        left = max(0, min(left, w - 1))
        right = max(0, min(right, w))
        top = max(0, min(top, h - 1))
        bottom = max(0, min(bottom, h))

        # Check for valid box
        if right <= left or bottom <= top:
            return False

        # Extract mask region for this bounding box
        submask = mask[top:bottom, left:right]

        if submask.size == 0:
            return False

        # Check if any part of the box is not in masked area
        if submask.max():  # If there are any unmasked pixels
            count = np.count_nonzero(submask == 0)  # Count unmasked pixels
            ratio = count / float(submask.size)
            if ratio > 0.0:
                x = 2
            return ratio > self.config.mask_ratio_threshold

        return True


def main():
    # Dataset configuration
    dataset = "hill_2min_2023-08"
    image_format = "JPG"

    # Create custom configuration with desired parameters
    config = IcebergDetectionConfig(
        dataset, image_format, masking=True, feature_extraction=True, num_workers=4, num_epochs=10, k_folds=5, patience=3, confidence_threshold=0.1
    )

    # Create detector instance, train the model and run inference
    detector = IcebergDetector(config=config)
    detector.train()
    detector.predict()


if __name__ == "__main__":
    main()