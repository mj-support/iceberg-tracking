import os
import copy
import cv2
import multiprocessing as mp
import torch
import time
import numpy as np
import pandas as pd
from datetime import timedelta
from PIL import Image
from sklearn.model_selection import KFold
from tqdm import tqdm
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Subset
from preprocessing import get_tiles_with_overlap
from utils.paths import DATA_DIR


class IcebergDataset(Dataset):
    def __init__(self, image_dir, image_format, det_file=None, transforms=None):
        """
        Dataset for iceberg detection that handles both training and inference modes.

        This dataset class loads images and their corresponding bounding box annotations
        for icebergs. In training mode (det_file provided), it reads annotations from a CSV file.
        In inference mode (det_file=None), it simply loads images from the directory.

        Args:
            image_dir (str): Directory containing the image files
            image_format (str): File formats of the images (file extension)
            det_file (str, optional): Path to detection/annotation file containing bounding boxes.
                                     If None, the dataset operates in inference-only mode.
            transforms (callable, optional): Image transformations to apply to each image
        """
        self.img_folder = image_dir
        self.image_format = image_format
        self.transforms = transforms
        self.detections = None
        self.unique_images = []

        if det_file and os.path.exists(det_file):
            # Read the detection file with pandas for training/validation mode
            column_names = ['image', 'iceberg_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                            'conf', 'unused_1', 'unused_2', 'unused_3']
            # Load CSV with predefined column names, containing bounding box annotations
            self.detections = pd.read_csv(det_file, names=column_names)
            # Extract unique image identifiers from annotations
            self.unique_images = self.detections['image'].unique()
        else:
            # Get filenames without extensions for all valid image files
            self.unique_images = [
                os.path.splitext(f)[0] for f in os.listdir(image_dir)
                if f.endswith(self.image_format)
            ]

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset by index.

        In training mode, returns the image and its corresponding target dictionary
        containing bounding boxes and labels. In inference mode, returns the image
        and a minimal target with only the image_id and file_name.

        Args:
            index (int): Index of the item to retrieve

        Returns:
            tuple: (image, target) where:
                - image is the transformed PIL image
                - target is a dictionary containing bounding boxes, labels, etc. in training mode,
                  or just image_id and file_name in inference mode

        Note:
            If an image has no valid bounding boxes in training mode,
            this method recursively tries the next image.
        """
        # Get image identifier from our list
        img_name = self.unique_images[index]
        img_file = os.path.join(self.img_folder, f"{img_name}{self.image_format}")

        # Load image and convert to RGB format
        img = Image.open(img_file).convert("RGB")
        # Create tensor for image ID
        image_id = torch.tensor([index])

        # For inference mode (no detections file) - return minimal target
        if self.detections is None:
            if self.transforms:
                img = self.transforms(img)
            return img, {"image_id": image_id, "file_name": img_name}

        # For training/validation mode - process annotations
        # Filter detections for this specific image
        img_detections = self.detections[self.detections['image'] == img_name]
        boxes = []
        labels = []

        # Process all detections for this image
        for _, det in img_detections.iterrows():
            xmin = det['bb_left']
            ymin = det['bb_top']
            width = det['bb_width']
            height = det['bb_height']

            # Convert from (x, y, width, height) to (x1, y1, x2, y2) format required by PyTorch
            xmax = xmin + width
            ymax = ymin + height

            # Filter out invalid bounding boxes with zero width or height
            if width > 0 and height > 0:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # Single class for iceberg (label 1, background is 0)

        # If no valid bounding boxes, we skip the image and try the next one
        # This prevents training issues with images that have no valid annotations
        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self.unique_images))

        # Convert lists to tensors with appropriate data types
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Calculate area of each bounding box (used by some loss functions)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # iscrowd indicates whether each object instance is a group of objects (0 = no)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # Create target dictionary in the format expected by Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'file_name': img_name  # Keep filename for reference
        }

        # Apply transformations if provided
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of unique images in the dataset
        """
        return len(self.unique_images)


class IcebergDetector:
    def __init__(self, dataset, image_format, tile=True, num_classes=2):
        """
        A comprehensive class for iceberg detection that handles both training and inference as well as postprocessing.

        This class provides methods for training models with k-fold cross-validation,
        early stopping, and inference with custom confidence threshold. It manages the
        model lifecycle including building, saving, loading, and using the model.

        Args:
            dataset (str): Name of the dataset, used for organizing files
            image_format (str): File formats of the images (file extension)
            tile (bool): Whether the images are split into tiles (default: True)
            num_classes (int): Number of classes including background (default: 2,
                               where 0 is background and 1 is iceberg)
        """
        self.dataset = dataset
        self.image_format = f".{image_format}"
        self.num_classes = num_classes
        self.tile = tile
        # Automatically use GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        # Create directory structure if not exists
        # Ensures all required folders are available for data organization
        os.makedirs(os.path.join(DATA_DIR, dataset, "models"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, dataset, "detections"), exist_ok=True)

        # Define all file and directory paths
        self.model_file = os.path.join(DATA_DIR, dataset, "models", "iceberg_detector.pth")
        self.images_dir = os.path.join(DATA_DIR, dataset, "images", "processed")
        self.annotations_file = os.path.join(DATA_DIR, dataset, "annotations", "gt.txt")
        if self.tile:
            self.detections_file = os.path.join(DATA_DIR, dataset, "detections", "det_tiles.txt")
            self.detections_file_merged = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
        else:
            self.detections_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")

    def train(self, k_folds=5, num_epochs=10, patience=3, batch_size=2, learning_rate=0.005):
        """
        Train the model using k-fold cross-validation with early stopping and timing metrics.

        This method implements a complete training pipeline with several advanced features:
        1. K-fold cross-validation to improve generalization
        2. Early stopping to prevent overfitting
        3. Comprehensive timing metrics for total and per-epoch progress
        4. Time remaining estimates based on running averages
        5. Best model selection based on validation loss

        The method tracks training across all folds, saving only the best-performing model
        based on validation loss. Progress and timing information is displayed throughout
        the training process.

        Args:
            k_folds (int): Number of folds for cross-validation, dividing the dataset
                           into k parts for rotating validation
            num_epochs (int): Maximum number of training epochs per fold
            patience (int): Number of consecutive epochs with no improvement before
                           triggering early stopping
            batch_size (int): Number of samples per batch for training and validation
            learning_rate (float): Learning rate for the optimizer
        """
        # Start timing the entire training process
        total_start_time = time.time()
        print(f"\nStarting training with {k_folds}-fold cross-validation")

        # Create complete dataset with transformations
        transformed_dataset = IcebergDataset(
            image_dir=self.images_dir,
            image_format=self.image_format,
            det_file=self.annotations_file,
            transforms=self._get_transform()
        )

        # Initialize k-fold cross-validation splitter with fixed random seed for reproducibility
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_val_loss_overall = float('inf')  # Track best loss across all folds
        best_model_state_overall = None  # Save best model state across all folds

        # Iterate through each fold in the cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(transformed_dataset)):
            # Start timing for this specific fold
            fold_start_time = time.time()
            print(f"\nFold {fold + 1}/{k_folds}")

            # Create training and validation subsets based on the fold split
            train_subset = Subset(transformed_dataset, train_idx)
            val_subset = Subset(transformed_dataset, val_idx)

            # Create data loaders with appropriate settings
            # The collate_fn is needed to handle variable-sized images and targets
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=4, collate_fn=lambda x: tuple(zip(*x))
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=4, collate_fn=lambda x: tuple(zip(*x))
            )

            # Initialize a fresh model for each fold to ensure independence
            model = self._build_model()

            # Configure optimizer with SGD, momentum and weight decay for regularization
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,  # Momentum helps escape local minima and accelerates convergence
                weight_decay=0.0005  # L2 regularization to prevent overfitting
            )

            # Setup for training loop with early stopping
            best_val_loss = float('inf')  # Track best loss within this fold
            best_model_state = None  # Save best model state within this fold
            epochs_no_improve = 0  # Counter for early stopping
            epoch_times = []  # List to store execution times for time estimation

            # Main training loop - iterate through epochs
            for epoch in range(num_epochs):
                # Start timing this epoch
                epoch_start_time = time.time()

                # Train for one epoch and evaluate on validation set
                train_loss = self._train_one_epoch(model, optimizer, train_loader)
                val_loss = self._evaluate(model, val_loader)

                # Calculate time taken for this epoch
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                epoch_times.append(epoch_duration)
                current_total_time, estimated_total_remaining, avg_time_per_epoch = (
                    self._calculate_training_progress(total_start_time, k_folds, fold, num_epochs, epoch)
                )

                # Print current epoch stats and global time estimates
                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                      f"Time: {timedelta(seconds=int(current_total_time))}<"
                      f"{timedelta(seconds=int(estimated_total_remaining))}, "
                      f"{timedelta(seconds=int(avg_time_per_epoch))}/Epoch")

                # Check for improvement in validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Deep copy to ensure we save the exact state without reference issues
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0  # Reset early stopping counter
                else:
                    epochs_no_improve += 1  # Increment early stopping counter

                # Early stopping check
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {epochs_no_improve} epochs without improvement")
                    break  # Exit the epoch loop

            # Calculate and print fold timing information
            fold_duration = time.time() - fold_start_time
            print(f"Fold {fold + 1} completed in {timedelta(seconds=int(fold_duration))}")

            # Update best model across all folds if this fold performed better
            if best_val_loss < best_val_loss_overall:
                best_val_loss_overall = best_val_loss
                best_model_state_overall = best_model_state
                print(f"New best model with validation loss: {best_val_loss_overall:.4f}")

        # Calculate and print total training time across all folds
        total_duration = time.time() - total_start_time
        print(f"\nTraining complete in {timedelta(seconds=int(total_duration))}")

        # Save the best model from all folds
        if best_model_state_overall:
            # Save the model state dictionary to the predefined path
            torch.save(best_model_state_overall, self.model_file)
            print(f"Best model saved with validation loss: {best_val_loss_overall:.4f}")
            print(f"Model saved to {self.model_file}")

    def predict(self, confidence_threshold=0.0):
        """
        Run inference on images, save detections to file and postprocess the results.

        This method loads a trained model, processes all images in the dataset directory,
        generates bounding box predictions, and saves them to a detection file in the
        required format for evaluation.

        Args:
            confidence_threshold (float): Minimum confidence score for detections to be included
                                         in the output. Higher values result in fewer but more
                                         confident detections. Default is 0.0 (include all).

        Returns:
            None: Results are saved to the detection file specified in the self.detections_files

        Raises:
            FileNotFoundError: If no trained model is found, this method suggests
                              training a model first.
        """
        print(f"\nStarting prediction with {confidence_threshold} confidence threshold")
        try:
            # Attempt to load the trained model
            model = self._load_model()
        except FileNotFoundError as e:
            print(e)
            print("Please train a model first or provide a pre-trained model.")
            return

        print(f"Running detection model on images in {self.images_dir}")
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()

        # Create a dataset for inference (no annotations needed)
        transformed_dataset = IcebergDataset(
            image_dir=self.images_dir,
            image_format=self.image_format,
            transforms=self._get_transform()
        )

        # Create dataloader for inference with batch size 1
        dataloader = DataLoader(
            transformed_dataset, batch_size=1, shuffle=False,
            num_workers=4, collate_fn=lambda x: tuple(zip(*x))
        )

        # Create progress bar for tracking
        progress_bar = tqdm(
            enumerate(dataloader),
            desc="Predict icebergs",
            total=len(dataloader),
            unit="image",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Prepare detection output file in the required format
        with open(self.detections_file, 'w') as det_file:
            for i, (images, targets) in progress_bar:
                # Get the single image and target from batch
                image = images[0].to(self.device)
                target = targets[0]
                img_name = target["file_name"]

                # Update progress bar description with current file
                progress_bar.set_description(f"Processing {img_name}")

                # Perform inference without gradient calculation
                with torch.no_grad():
                    prediction = model([image])

                # Extract predictions
                boxes = prediction[0]['boxes'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()

                # Frame ID is the image name without leading zeros (per format requirements)
                frame_id = img_name.lstrip('0')
                object_id = 1  # Initial object ID for each image

                # Process each detection and write to file if confidence exceeds threshold
                for box, score in zip(boxes, scores):
                    if score > confidence_threshold:
                        # Extract box coordinates
                        xmin, ymin, xmax, ymax = box
                        # Convert to width and height format
                        width, height = xmax - xmin, ymax - ymin
                        # Placeholder values for 3D coordinates (not used)
                        x, y, z = -1, -1, -1

                        # Write detection in required format:
                        # frame_id, object_id, x, y, width, height, score, x_3d, y_3d, z_3d
                        det_file.write(
                            f"{frame_id},{object_id},{xmin},{ymin},{width},{height},{score},{x},{y},{z}\n"
                        )
                        object_id += 1  # Increment object ID for next detection

        print(f"Detections saved to {self.detections_file}")
        print(f"\nStarting postprocessing {self.detections_file}")
        self.postprocess()

    def postprocess(self):
        """
        Performs postprocessing on detection results.

        This method handles the final processing steps after object detection:
        1. Removes masked (filtered out) detections
        2. Merges tiled detections if tiling was used during preprocessing
        3. Sort all detections by image name
        """
        # Remove detections that were marked for filtering
        self._remove_masked_detections()

        # If images were processed as tiles, merge the detections back to original image coordinates
        if self.tile:
            self._merge_tiles()
            self._sort(self.detections_file_merged)
        else:
            self._sort(self.detections_file)

        print("Finished postprocessing")

    def _get_transform(self):
        """
        Get image transformation function for preprocessing input images.

        This method defines the transformations applied to each image before
        feeding it to the model. Currently, it only converts the image to a
        PyTorch tensor and normalizes pixel values to [0,1].

        Returns:
            function: A callable transformation function that takes an image
                     and returns the transformed version.
        """

        def transform(image):
            # Convert PIL image or numpy.ndarray to tensor and normalize to [0,1]
            image = F.to_tensor(image)
            return image

        return transform

    def _build_model(self):
        """
        Create and return the Faster R-CNN model architecture.

        This method initializes a Faster R-CNN model with ResNet50 backbone and
        Feature Pyramid Network (FPN). It adapts the pretrained model for our
        specific number of classes by replacing the box predictor head.

        Returns:
            torch.nn.Module: Configured Faster R-CNN model moved to the appropriate device.
        """
        # Initialize with pretrained weights for feature extraction
        model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one for our specific number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Move model to the specified device (CPU or GPU)
        return model.to(self.device)

    def _load_model(self):
        """
        Load a previously trained model for inference.

        This method checks if a model instance exists, creates one if needed,
        and loads the trained weights from disk. It then puts the model in
        evaluation mode for inference.

        Returns:
            torch.nn.Module: The loaded model ready for inference.

        Raises:
            FileNotFoundError: If no model file exists at the specified path.
        """
        # Initialize model if not already done
        if self.model is None:
            self.model = self._build_model()

        # Check if model file exists and load weights
        if os.path.exists(self.model_file):
            # Load the saved state dictionary, handling device mapping
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            # Set model to evaluation mode
            self.model.eval()
            print(f"Model loaded from {self.model_file}")
        else:
            raise FileNotFoundError(f"No trained model found at {self.model_file}")

        return self.model

    def _train_one_epoch(self, model, optimizer, data_loader):
        """
        Train the model for one complete epoch.

        This method runs one training epoch, processing all batches in the data loader.
        For each batch, it computes the forward pass, calculates losses, and updates
        model parameters through backpropagation.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
            data_loader (torch.utils.data.DataLoader): DataLoader providing training batches.

        Returns:
            float: Average loss value for the epoch.
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        model.train()
        running_loss = 0.0

        # Iterate through all batches in the data loader
        for i, (images, targets) in enumerate(data_loader):
            # Move images and targets to the appropriate device
            images = list(image.to(self.device) for image in images)
            # Filter out non-tensor elements like 'file_name' and move tensors to device
            targets = [{k: v.to(self.device) for k, v in t.items() if k != 'file_name'} for t in targets]

            # Forward pass: compute predictions and losses
            # Faster R-CNN returns a dict of losses when targets are provided
            loss_dict = model(images, targets)
            # Sum all individual losses (classification, regression, etc.)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            losses.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Accumulate batch loss
            running_loss += losses.item()

        # Return average loss over all batches
        return running_loss / len(data_loader)

    def _calculate_training_progress(self, total_start_time, k_folds, fold, num_epochs, epoch):
        """
        Calculate global training progress and time estimation metrics across all folds.

        This method computes the current runtime, estimates the remaining time, and
        calculates the average epoch execution time based on the overall training progress.
        These metrics provide a global perspective on training status rather than fold-specific
        information.

        Args:
            total_start_time (float): Unix timestamp when the entire training process started
            k_folds (int): Total number of folds in cross-validation
            fold (int): Current fold index (0-based)
            num_epochs (int): Maximum number of epochs per fold
            epoch (int): Current epoch index within the current fold (0-based)

        Returns:
            tuple: A tuple containing:
                - current_total_time (float): Elapsed time since training started (in seconds)
                - estimated_total_remaining (float): Estimated time to complete all remaining epochs (in seconds)
                - avg_time_per_epoch (float): Average time per epoch across all completed epochs (in seconds)
        """
        # Calculate elapsed time since the start of the entire training process
        current_total_time = time.time() - total_start_time

        # Calculate the total number of epochs across all folds
        total_epochs = k_folds * num_epochs

        # Calculate the global epoch number (1-based) across all folds
        # This represents which epoch we're on if we consider all folds sequentially
        current_epoch_global = fold * num_epochs + (epoch + 1)

        # Number of epochs we've completed so far across all folds
        epochs_completed = current_epoch_global

        # Calculate the average time taken per epoch based on all epochs completed so far
        # This provides a more stable estimate as training progresses
        avg_time_per_epoch = current_total_time / epochs_completed

        # Calculate how many epochs remain across all folds
        remaining_epochs = total_epochs - epochs_completed

        # Estimate the total time remaining for the entire training process
        # based on average time per epoch and number of remaining epochs
        estimated_total_remaining = remaining_epochs * avg_time_per_epoch

        return current_total_time, estimated_total_remaining, avg_time_per_epoch

    def _evaluate(self, model, data_loader):
        """
        Evaluate model performance on validation or test data.

        This method computes the model's loss on the provided data without
        updating model parameters. This is used for validation during training
        or final evaluation.

        Note: We use train mode with gradients enabled to calculate losses,
        but don't update parameters. This ensures consistency with the loss
        calculation during training.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): DataLoader providing validation batches.

        Returns:
            float: Average validation loss value.
        """
        # Use train mode to ensure loss calculation matches training
        # This might seem counterintuitive but ensures loss computation uses the same settings
        model.train()
        running_val_loss = 0.0

        # Enable gradients for loss calculation but don't update model
        with torch.set_grad_enabled(True):
            # Iterate through all batches in the data loader
            for images, targets in data_loader:
                # Move images and targets to the appropriate device
                images = list(image.to(self.device) for image in images)
                # Filter out non-tensor elements and move tensors to device
                targets = [{k: v.to(self.device) for k, v in t.items() if k != 'file_name'} for t in targets]

                # Forward pass only to calculate losses
                loss_dict = model(images, targets)
                # Sum all individual losses
                losses = sum(loss for loss in loss_dict.values())
                # Accumulate batch loss
                running_val_loss += losses.item()

        # Return average loss over all batches
        return running_val_loss / len(data_loader)

    def _remove_masked_detections(self):
        """
        Remove detections that are primarily in masked (black) areas of the images.

        This method uses parallel processing to efficiently handle large numbers of detections.
        It checks each bounding box against the corresponding image to determine if it primarily
        contains black pixels (mask area), and writes a new detection file without the masked detections.

        Note:
            This method creates a temporary file and then replaces the original detection file.
        """
        import multiprocessing as mp
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor, as_completed

        black_threshold = 0.8
        print(f"Removing detections in masked areas (black threshold: {black_threshold})")

        # Create a temporary file for filtered detections
        temp_file = self.detections_file + ".temp"

        try:
            # First read all detections and prepare data
            all_detections = []
            with open(self.detections_file, 'r') as in_file:
                for line in in_file:
                    parts = line.strip().split(',')
                    if len(parts) < 7:  # Ensure we have at least the essential fields
                        all_detections.append(
                            (line, None, None, None, None, None, self.images_dir, self.image_format, black_threshold))
                        continue

                    # Extract bounding box information
                    frame_id = parts[0]
                    x, y = float(parts[2]), float(parts[3])
                    width, height = float(parts[4]), float(parts[5])
                    all_detections.append(
                        (line, frame_id, x, y, width, height, self.images_dir, self.image_format, black_threshold))

            total_detections = len(all_detections)
            print(f"Loaded {total_detections} detections for processing")

            # Process detections in parallel
            num_workers = min(mp.cpu_count(), 16)  # Use up to 16 workers or max CPU cores
            results = []
            masked_detections = 0

            print(f"Processing with {num_workers} parallel workers")

            # Create progress bar for tracking
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all detection processing tasks
                futures = [executor.submit(_process_masked_detection, detection_data)
                           for detection_data in all_detections]

                # Track progress with tqdm
                with tqdm(
                        total=total_detections,
                        desc="Processing detections",
                        unit="detection",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ) as progress_bar:
                    for future in as_completed(futures):
                        progress_bar.update(1)
                        line, is_masked = future.result()
                        if is_masked:
                            masked_detections += 1
                        else:
                            results.append(line)

            # Write filtered results to file
            with open(temp_file, 'w') as out_file:
                for line in results:
                    out_file.write(line)

            # Replace original file with filtered file
            os.replace(temp_file, self.detections_file)

            # Print statistics
            print(
                f"Removed {masked_detections} of {total_detections} detections ({masked_detections / total_detections * 100:.1f}%) in masked areas")

        except Exception as e:
            print(f"Error processing detection file: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def _merge_tiles(self):
        """
        Merges detection results from image tiles back into original image coordinates.

        When large images are processed as tiles (smaller segments), this method:
        1. Adjusts bounding box coordinates based on tile positions in original image
        2. Ensures detection IDs remain unique when merging tiles
        3. Writes merged detections to a new output file

        The method reads the original detection file line by line, transforms coordinates,
        and writes to a merged detection file with updated values.
        """
        print("Merge tiles to reconstruct original images")

        # Get information about how images were tiled, including overlap regions
        tiles = get_tiles_with_overlap(self.dataset)

        # Dictionary to track existing object IDs for each image to avoid duplicates
        existing_ids = {}

        # Process the detection file and create a new merged output file
        with open(self.detections_file, "r") as infile, open(self.detections_file_merged, "w") as outfile:
            for line in infile:
                # Parse the CSV line into components
                parts = line.strip().split(",")

                # Skip processing if line doesn't have enough fields (likely malformed)
                if len(parts) < 10:
                    continue

                # Extract key information from the detection line
                image_name = parts[0]  # Image filename including tile indicator
                tile_suffix = image_name[-1]  # The tile identifier (last character)
                original_id = int(parts[1])  # Detection ID within the tile

                # Only process if this is a recognized tile
                if tile_suffix in tiles:
                    # Get tile position in original image for coordinate transformation
                    x_offset = tiles[tile_suffix]["xmin"]
                    y_offset = tiles[tile_suffix]["ymin"]

                    # Transform bounding box coordinates from tile space to original image space
                    x = float(parts[2]) + x_offset
                    y = float(parts[3]) + y_offset
                    width = float(parts[4])  # Width doesn't need adjustment
                    height = float(parts[5])  # Height doesn't need adjustment

                    # Get the original image name by removing tile suffix (e.g., "_A")
                    clean_image_name = image_name[:-2]

                    # Initialize tracking for this image if not already done
                    if clean_image_name not in existing_ids:
                        existing_ids[clean_image_name] = set()

                    # Handle ID collision - ensure unique object IDs across tiles
                    if original_id in existing_ids[clean_image_name]:
                        # Find the next available ID if this one is already used
                        new_id = 1
                        while new_id in existing_ids[clean_image_name]:
                            new_id += 1
                    else:
                        # Use the original ID if no collision
                        new_id = original_id

                    # Record this ID as now used for this image
                    existing_ids[clean_image_name].add(new_id)

                    # Format and write the new merged detection entry
                    # Format: image_name,id,x,y,width,height,confidence,class,other_fields...
                    new_line = f"{clean_image_name},{new_id},{x},{y},{width},{height},{parts[6]},{parts[7]},{parts[8]},{parts[9]}\n"
                    outfile.write(new_line)

        print(f"Detections of the merged tiles saved to {self.detections_file_merged}")

    def _sort(self, det_file):
        """
        Sort detection records in a file by filename and object ID.

        This method reads a detection file containing comma-separated values,
        sorts the lines first by filename and then by object ID, and writes
        the sorted lines back to the same file.

        Args:
            det_file: Path to the detection file containing comma-separated records.
                Each line is expected to have at least two fields: filename and object ID (integer).
        """
        with open(det_file, 'r') as f:
            lines = f.readlines()

        parsed = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                filename = parts[0]
                try:
                    obj_id = int(parts[1])
                except ValueError:
                    obj_id = float('inf')  # fallback for bad data
                parsed.append((filename, obj_id, line))

        parsed.sort(key=lambda x: (x[0], x[1]))
        sorted_lines = [line for _, _, line in parsed]

        with open(det_file, "w") as f:
            f.writelines(sorted_lines)


def _process_masked_detection(args):
    """
    Process a single detection to determine if it's in a masked area.

    Args:
        args: Tuple containing:
            - line: Original detection line
            - frame_id: ID of the frame
            - x, y, width, height: Bounding box coordinates
            - images_dir: Directory containing images
            - image_format: Format of image files (e.g., '.jpg')
            - black_threshold: Threshold for determining masked detections

    Returns:
        Tuple: (line, is_masked)
    """
    line, frame_id, x, y, width, height, images_dir, image_format, black_threshold = args

    # Handle cases where we don't have valid detection data
    if frame_id is None:
        return line, False

    # Construct image path
    img_name = str(frame_id).zfill(6) + image_format
    img_file = os.path.join(images_dir, img_name)

    # Check if image exists
    if not os.path.exists(img_file):
        return line, False

    # Load image (using OpenCV for efficient image processing)
    img = cv2.imread(img_file)
    if img is None:
        return line, False

    # Ensure coordinates are within image bounds
    img_height, img_width = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_width, int(x + width))
    y2 = min(img_height, int(y + height))

    # Skip invalid bounding boxes
    if x1 >= x2 or y1 >= y2:
        return line, False

    # Extract bounding box region
    bbox_region = img[y1:y2, x1:x2]

    # Count black pixels ([0,0,0]) in the bounding box
    black_pixel_threshold = 10  # Tolerance for nearly black pixels
    black_pixels = np.sum(np.all(bbox_region < black_pixel_threshold, axis=2))
    total_pixels = bbox_region.shape[0] * bbox_region.shape[1]

    # Calculate ratio of black pixels
    black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

    # If black ratio is above threshold, mark for removal
    return line, black_ratio >= black_threshold


def main():
    # Create detector instance
    dataset = "fjord_2min_2023-08"

    detector = IcebergDetector(dataset, image_format="JPG")

    # Train the model
    detector.train(k_folds=5, num_epochs=10, patience=3)

    # Run inference
    detector.predict(confidence_threshold=0.0)

    detector.postprocess()


if __name__ == "__main__":
    main()