import cv2
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from segment_anything import SamPredictor, sam_model_registry
import torch
from tqdm import tqdm
import urllib.request

from utils.helpers import PROJECT_ROOT, load_icebergs_by_frame, get_sequences, get_image_ext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Iceberg Tracking Visualization Module

This module provides visualization capabilities for iceberg tracking data create 
annotated images and videos for analysis, validation and presentation purposes. 
It supports multiple annotation sources and flexible rendering options.

Key Features:
    1. Multi-Source Annotation Support (Ground truth, detections, tracking, eval)
    2. Flexible Visualization Options (bounding boxes, IDs, contours, masks)
    3. Video Generation
    4. Advanced Segmentation
    5. Consistent Color Mapping
"""


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class VisualizationConfig:
    """
    Configuration for iceberg tracking visualization and video generation.

    This dataclass centralizes all parameters for creating annotated images and videos
    from iceberg tracking data. It supports multiple annotation sources and flexible
    rendering options for analysis, validation, and presentation.

    Configuration Categories:
        - Data: Dataset and annotation source selection
        - Processing: Frame range and display options
        - Annotation: Visual elements to render (IDs, boxes, contours, masks)
        - Video: Output video settings

    Attributes:
        dataset (str): Name/path of dataset to visualize
            Examples: "hill/test", "columbia/ice_melange"
            Must contain images/ directory and annotation files

        annotation_source (str): Source of annotation data
            Options:
                - "tracking": Complete tracking results (default)
                - "detections": Raw detector outputs
                - "ground_truth": Manual annotations
                - "eval": Evaluation results
            Default: "tracking"

        start_index (int): Starting frame index for processing. Default: 0 (start from beginning)

        seq_length_limit (int | None): Maximum number of frames to process. Default: None (process all)

        show_images (bool): Display images during processing. Default: False

        draw_ids (bool): Draw iceberg ID numbers. Default: True
        draw_boxes (bool): Draw bounding boxes. Default: True
        draw_contours (bool): Draw SAM segmentation contours. Default: False
        draw_masks (bool): Draw semi-transparent segmentation masks. Default: False
        fps (int): Video frame rate (frames per second). Default: 7

    Performance Notes:
        - draw_contours or draw_masks: Requires SAM model (~400MB download on first use)
        - SAM processing leads to major peformance decrease compare to annotate only bounding boxes / ID

    Workflow:
        1. Create config with desired options
        2. Initialize Visualizer with config
        3. Call annotate_icebergs() to process frames
        4. Call render_video() to create MP4

    Examples:
        >>> # Basic visualization with IDs and boxes
        >>> config = VisualizationConfig(
        ...     dataset="hill/test",
        ...     draw_ids=True,
        ...     draw_boxes=True
    """
    # Data configuration
    dataset: str

    # General configurations
    annotation_source: str = "tracking"
    start_index: int = 0
    seq_length_limit: int | None = None
    show_images: bool = False

    # Annotation configuration
    draw_ids: bool = True
    draw_boxes: bool = True
    draw_contours: bool = False
    draw_masks: bool = False

    # Video configurations
    fps: int = 7


# ============================================================================
# MAIN VISUALIZER CLASS
# ============================================================================

class Visualizer:
    """
    Main orchestrator for iceberg tracking visualization and video generation.

    This class provides a complete pipeline for annotating images with iceberg
    tracking information and generating videos from annotated sequences. It
    supports multiple annotation sources and flexible rendering options.

    The visualizer can work with three types of annotation sources:
    1. Ground Truth: Manual annotations for training/validation
    2. Detections: Raw detector outputs (Faster R-CNN)
    3. Tracking: Complete tracking results (MOT algorithm)

    Visualization Options:
        - Bounding boxes: Rectangular regions around icebergs
        - IDs: Unique identifiers for tracking continuity
        - Contours: Precise boundaries from SAM segmentation
        - Masks: Semi-transparent overlays showing iceberg regions

    Workflow:
        1. Initialize with dataset and configuration
        2. Call annotate_icebergs() to process frames
        3. Optionally call render_video() to create MP4

    Methods:
        annotate_icebergs(): Process frames and add annotations
        render_video(): Compile annotated images into video
        _get_selection(): Load and validate data
        _get_sam_predictor(): Initialize SAM model
        _map_icebergs(): Core annotation loop
        _get_object_color(): Generate consistent colors
        _segment_icebergs(): SAM segmentation
        _export_images(): Save annotated frames
    """

    def __init__(self, config: VisualizationConfig):
        """
        Initialize the Visualizer with dataset and configuration.

        Sets up paths, validates annotation source, and configures processing
        parameters for the visualization pipeline.
        """
        self.dataset = config.dataset
        self.annotation_source = config.annotation_source
        self.show_images = config.show_images
        self.start_index = config.start_index
        self.seq_length_limit = config.seq_length_limit
        self.config = config

        # Determine device for SAM model (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Validate annotation source
        valid_sources = ["ground_truth", "detections", "tracking", "eval"]
        if self.annotation_source not in valid_sources:
            raise ValueError(
                f"Invalid annotation source '{self.annotation_source}'. "
                f"Must be one of: {', '.join(valid_sources)}"
            )

    def annotate_icebergs(self):
        """
        Annotate iceberg images with various visualization options.

        This is the main processing method that loads images, applies requested
        annotations, and saves the results. It processes all sequences in the
        dataset within the specified frame range.

        Visualization Options:
            draw_ids: Display numeric IDs above each iceberg
            draw_boxes: Display bounding boxes around icebergs
            draw_contours: Display precise boundaries using SAM
            draw_masks: Display semi-transparent segmentation masks

        Performance:
            - SAM segmentation adds significant processing time
        """
        logger.info("Starting iceberg annotation with configuration:")
        logger.info(f"  draw_ids = {self.config.draw_ids}")
        logger.info(f"  draw_boxes = {self.config.draw_boxes}")
        logger.info(f"  draw_contours = {self.config.draw_contours}")
        logger.info(f"  draw_masks = {self.config.draw_masks}")

        # Get all sequences in the dataset
        sequences = get_sequences(self.dataset)

        for sequence_name, paths in sequences.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing sequence: {sequence_name}")
            logger.info(f"{'=' * 60}")

            # Load and validate annotation data for selected frames
            icebergs_by_frame, image_ext = self._get_selection(output_type="image", paths=paths)

            # Initialize SAM predictor if contours or masks are needed
            if self.config.draw_contours or self.config.draw_masks:
                sam_predictor = self._get_sam_predictor()
            else:
                sam_predictor = None

            # Process all frames and generate annotated images
            images = self._map_icebergs(
                icebergs_by_frame,
                paths,
                image_ext,
                sam_predictor
            )

            logger.info("Annotation complete.")

            # Save the annotated images
            self._export_images(icebergs_by_frame, images, paths)

    def render_video(self):
        """
        Create an MP4 video from annotated images.
        """
        logger.info(f"Starting video rendering at {self.config.fps} fps...")

        sequences = get_sequences(self.dataset)

        for sequence_name, paths in sequences.items():
            logger.info(f"\nProcessing sequence: {sequence_name}")

            # Set up video output directory and filename
            base_path = str(paths["images"]).split("/images")[0]
            video_dir = os.path.join(base_path, "visualizations", "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_name = f"{self.annotation_source}.mp4"
            video_path = os.path.join(video_dir, video_name)

            # Verify that annotated images exist
            image_dir = os.path.join(base_path, "visualizations", self.annotation_source)
            if not os.path.exists(image_dir):
                raise FileNotFoundError(
                    f"Annotated images directory not found: {image_dir}\n"
                    f"Please run annotate_icebergs() before render_video()."
                )

            # Get list of images to include in video
            images = self._get_selection(output_type="video", paths=paths)

            # Get video dimensions from first image
            first_image_path = os.path.join(image_dir, images[0])
            first_image = cv2.imread(first_image_path)
            if first_image is None:
                raise FileNotFoundError(f"Could not read first image: {first_image_path}")

            height, width = first_image.shape[:2]  # Remove the underscore for channels
            logger.info(f"Video dimensions: {width}x{height}")
            logger.info(f"Number of frames: {len(images)}")

            # Try different codecs in order of preference
            codecs = [
                ('avc1', '.mp4'),  # H.264 - most compatible
                ('mp4v', '.mp4'),  # MPEG-4
                ('XVID', '.avi'),  # Xvid fallback
            ]

            video_writer = None
            for codec, ext in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_path = video_path.replace('.mp4', ext)
                    video_writer = cv2.VideoWriter(test_path, fourcc, self.config.fps, (width, height))

                    # Test if writer was created successfully
                    if video_writer.isOpened():
                        video_path = test_path
                        logger.info(f"Using codec: {codec}")
                        break
                    else:
                        video_writer.release()
                        video_writer = None
                except Exception as e:
                    logger.warning(f"Codec {codec} failed: {e}")
                    continue

            if video_writer is None:
                raise RuntimeError("Failed to initialize video writer with any codec")

            # Write each frame to the video with progress tracking
            logger.info("Writing frames to video...")
            failed_frames = 0

            for frame_name in tqdm(images, desc="Writing video", unit="frame"):
                image_path = os.path.join(image_dir, frame_name)
                img = cv2.imread(image_path)

                if img is None:
                    logger.warning(f"Could not read frame: {frame_name}")
                    failed_frames += 1
                    continue

                # Ensure frame has correct dimensions
                if img.shape[:2] != (height, width):
                    logger.warning(f"Frame {frame_name} has wrong dimensions {img.shape[:2]}, resizing...")
                    img = cv2.resize(img, (width, height))

                # Ensure frame is in correct format (uint8, 3 channels)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Write frame
                video_writer.write(img)

            # Clean up and report completion
            video_writer.release()

            if failed_frames > 0:
                logger.warning(f"Failed to write {failed_frames} frames")

            logger.info("Video rendering complete.")
            logger.info(f"Video saved to: {video_path}")

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _get_selection(self, output_type, paths):
        """
        Load and validate data for the selected frame range.

        This method handles data loading differently depending on whether
        we're preparing for image annotation or video rendering. It ensures
        all required data exists before processing begins.

        Args:
            output_type (str): Type of output being prepared
                - "image": Preparing for annotation
                - "video": Preparing for video compilation
            paths (dict): Paths dictionary from get_sequences()

        Returns:
            For "image":
                tuple: (icebergs_by_frame, image_ext)
                    - icebergs_by_frame (dict): Frame name -> iceberg data
                    - image_ext (str): Image file extension (e.g., "jpg")
            For "video":
                list: Filenames of annotated images
        """
        # Get sorted list of all images in the directory
        image_ext = get_image_ext(paths["images"])
        images = [f for f in os.listdir(paths["images"]) if f.endswith(image_ext)]
        images.sort()

        # Slice to get only the requested range
        if self.seq_length_limit is None:
            end_frame = len(images)
        else:
            end_frame = min(self.start_index + self.seq_length_limit, len(images))

        images = images[self.start_index:end_frame]

        logger.info(f"Selected {len(images)} frames (index {self.start_index} to {self.start_index + len(images) - 1})")

        if output_type == "image":
            # Image annotation workflow: load and validate annotation data

            # Load iceberg annotations from file
            icebergs_by_frame = load_icebergs_by_frame(paths[self.annotation_source])
            sliced_icebergs_by_frame = {}

            # Validate that annotation data exists for each selected frame
            missing_count = 0
            for image in images:
                # Remove file extension to match annotation file format
                image_key = image.split("." + image_ext)[0]

                if image_key not in icebergs_by_frame:
                    # Log warning but continue processing other frames
                    logger.warning(
                        f"Missing annotation data for frame '{image}' in {self.annotation_source}"
                    )
                    missing_count += 1
                else:
                    # Add valid annotation data to the selection
                    sliced_icebergs_by_frame[image_key] = icebergs_by_frame[image_key]

            if missing_count > 0:
                logger.warning(f"Skipped {missing_count} frames due to missing annotations")

            return sliced_icebergs_by_frame, image_ext

        elif output_type == "video":
            # Video rendering workflow: get list of annotated images
            base_path = str(paths["images"]).split("/images")[0]
            image_dir = os.path.join(base_path, "visualizations", self.annotation_source)

            # Get list of available annotated output images
            annotated_images = sorted([
                f for f in os.listdir(image_dir)
                if f.lower().endswith("jpg")
            ])

            logger.info(f"Found {len(annotated_images)} annotated images")
            return annotated_images

    def _get_sam_predictor(self):
        """
        Initialize and return a SAM (Segment Anything Model) predictor.

        SAM is a powerful segmentation model that can generate precise masks
        for objects given simple prompts like bounding boxes. This method
        handles the complete setup:
        1. Download model weights if not present (~375MB)
        2. Load model architecture
        3. Move model to appropriate device (GPU/CPU)
        4. Create predictor instance

        Returns:
            SamPredictor: Configured predictor ready for segmentation
        """
        logger.info("Initializing SAM model...")

        # Define SAM model paths and download URL
        sam_weights_filename = "sam_vit_b_01ec64.pth"
        sam_weights_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(sam_weights_dir, exist_ok=True)
        sam_weights_path = os.path.join(sam_weights_dir, sam_weights_filename)
        sam_weights_url = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_weights_filename

        # Download weights if they don't exist locally
        if not os.path.exists(sam_weights_path):
            logger.info(f"SAM weights not found. Downloading to: {sam_weights_path}")
            logger.info(f"Download size: ~375MB (this may take a few minutes)")
            try:
                urllib.request.urlretrieve(sam_weights_url, sam_weights_path)
                logger.info("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download SAM weights: {e}")

        # Load and configure the SAM model
        logger.info(f"Loading SAM model on device: {self.device}")
        sam = sam_model_registry["vit_b"](checkpoint=sam_weights_path)
        sam.to(device=self.device)
        logger.info("SAM model loaded successfully.")

        return SamPredictor(sam)

    def _map_icebergs(self, icebergs_by_frame, paths, image_ext, sam_predictor=None):
        """
        Process all frames and add iceberg annotations to images.

        This is the core processing loop that iterates through all frames and
        applies the requested annotations to each iceberg. It maintains consistent
        colors across frames for tracking clarity.

        Args:
            icebergs_by_frame (dict): Frame name -> iceberg data mapping
            paths (dict): Paths dictionary from get_sequences()
            image_ext (str): Image file extension (e.g., "jpg")
            sam_predictor (SamPredictor | None): SAM predictor instance

        Returns:
            list: List of annotated images as numpy arrays (BGR format)
        """
        images_with_mappings = []
        colormap = {}  # Store consistent colors for each iceberg ID

        # Calculate total number of icebergs for progress bar
        total_entries = sum(len(icebergs) for icebergs in icebergs_by_frame.values())

        # Initialize progress bar
        progress_bar = tqdm(
            total=total_entries,
            desc="Processing icebergs",
            unit="iceberg",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Process each frame
        for frame_name, icebergs in icebergs_by_frame.items():
            progress_bar.set_description(f"Processing {frame_name}.{image_ext}")

            # Load the raw image
            image_path = paths["images"] / (frame_name + "." + image_ext)
            img = cv2.imread(str(image_path))
            if img is None:
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Prepare SAM predictor if needed for segmentation
            if self.config.draw_contours or self.config.draw_masks:
                # SAM expects RGB, OpenCV loads as BGR
                outline_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sam_predictor.set_image(outline_img)

            # Process each iceberg in the current frame
            for iceberg_id, iceberg_data in icebergs.items():
                progress_bar.update(1)

                # Get consistent color for this iceberg ID
                color = self._get_object_color(iceberg_id, colormap)

                # Extract bounding box coordinates
                x1, y1, w, h = iceberg_data['bbox']
                x2 = x1 + w  # Right edge
                y2 = y1 + h  # Bottom edge

                # Draw iceberg ID text if requested
                if self.config.draw_ids:
                    # Position text above bounding box (with minimum y to stay on screen)
                    text_y = max(int(y1) - 10, 20)
                    cv2.putText(
                        img,
                        str(iceberg_id),
                        (int(x1), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,  # Font scale
                        color,
                        2  # Thickness
                    )

                # Draw bounding box if requested
                if self.config.draw_boxes:
                    cv2.rectangle(
                        img,
                        (int(x1), int(y1)),  # Top-left corner
                        (int(x2), int(y2)),  # Bottom-right corner
                        color,
                        3  # Thickness
                    )

                # Generate and draw segmentation if requested
                if self.config.draw_contours or self.config.draw_masks:
                    # Run SAM segmentation with bounding box as prompt
                    contours, mask, score = self._segment_icebergs(
                        sam_predictor,
                        np.array([x1, y1, x2, y2])
                    )

                    # Apply semi-transparent mask overlay if requested
                    if self.config.draw_masks:
                        mask_alpha = 0.5  # 50% transparency
                        # Blend original image with solid color using mask
                        img[mask] = (
                                (1 - mask_alpha) * img[mask] +
                                mask_alpha * np.array(color)
                        ).astype(np.uint8)

                    # Draw contour outline if requested
                    if self.config.draw_contours:
                        cv2.drawContours(
                            img,
                            contours,
                            -1,  # Draw all contours
                            color,
                            2  # Thickness
                        )

            # Collect annotated image
            images_with_mappings.append(img)

        progress_bar.close()
        return images_with_mappings

    def _get_object_color(self, object_id, colormap):
        """
        Get a consistent, deterministic color for an object ID.

        This method ensures that the same object ID always gets the same color
        across all frames, making it easy to visually track objects through
        time. Colors are generated using seeded random numbers for perfect
        reproducibility.

        Args:
            object_id (int): Unique identifier for the object
            colormap (dict): Dictionary storing ID → color mappings

        Returns:
            tuple: BGR color tuple (B, G, R) for OpenCV
                - Values in range [0, 255]
                - OpenCV uses BGR format, not RGB
        """
        if object_id not in colormap:
            # Use object ID as seed for deterministic color generation
            np.random.seed(object_id)
            # Generate random BGR color
            color = tuple(map(int, np.random.randint(0, 255, size=3)))
            # Cache for future use
            colormap[object_id] = color

        return colormap[object_id]

    def _segment_icebergs(self, sam_predictor, input_box):
        """
        Generate precise segmentation mask and contours using SAM.

        This method uses the Segment Anything Model (SAM) to generate a precise
        segmentation mask for an iceberg, given its bounding box as a prompt.
        SAM can segment objects much more accurately than simple bounding boxes.

        Args:
            sam_predictor (SamPredictor): Configured SAM predictor
                - Must have image set via set_image()
            input_box (np.ndarray): Bounding box [x1, y1, x2, y2]
                - Top-left and bottom-right corners
                - Pixel coordinates

        Returns:
            tuple: (contours, mask, score)
                - contours: OpenCV contours for drawing
                    - List of numpy arrays or single array
                    - Can be passed directly to cv2.drawContours
                - mask: Binary segmentation mask
                    - Boolean array same size as image
                    - True inside iceberg, False outside
                - score: Confidence score from SAM
                    - Float in range [0, 1]
                    - Higher = more confident
        """
        # Run SAM prediction using bounding box as prompt
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,  # Not using point prompts
            point_labels=None,  # Not using point prompts
            box=input_box[None, :],  # Add batch dimension [1, 4]
            multimask_output=False  # Return single best mask
        )

        # Select the best mask based on confidence score
        # Filter to valid masks (score >= 0)
        valid_masks = scores >= 0
        if not valid_masks.any():
            # Fallback: use first mask if none are valid
            best_idx = 0
        else:
            # Select mask with highest score
            best_idx = np.argmax(scores[valid_masks])

        mask = masks[valid_masks][best_idx]
        score = scores[valid_masks][best_idx]

        # Convert binary mask to uint8 format for contour extraction
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Extract contours from mask
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress contours
        )

        # Select the largest contour if multiple are found
        # (SAM sometimes generates small spurious regions)
        if len(contours) > 1:
            # Compare by number of points (approximation of size)
            contours = [max(contours, key=lambda c: c.shape[0])]

        return contours, mask, score

    def _export_images(self, icebergs_by_frame, images, paths):
        """
        Save annotated images to the output directory.

        This method handles the final step of the annotation pipeline: saving
        all processed images to disk. It creates the necessary directory structure
        and optionally displays images using matplotlib.

        Args:
            icebergs_by_frame (dict): Frame names to iceberg data
                - Used to get frame names in correct order
            images (list): Annotated images as numpy arrays
                - Must match order of icebergs_by_frame
                - BGR format (OpenCV)
            paths (dict): Paths dictionary from get_sequences()
        """
        # Create output directory structure
        base_path = str(paths["images"]).split("/images")[0]
        image_dir = os.path.join(base_path, "visualizations", self.annotation_source)
        os.makedirs(image_dir, exist_ok=True)

        logger.info(f"Saving annotated images to: {image_dir}")

        # Save each annotated image
        for index, frame_name in enumerate(icebergs_by_frame):
            img = images[index]

            # Convert BGR to RGB for display purposes
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save the image (OpenCV uses BGR format)
            img_name = f"{frame_name}.jpg"
            output_path = os.path.join(image_dir, img_name)
            cv2.imwrite(output_path, img)

            # Display image if requested
            if self.show_images:
                plt.figure(figsize=(10, 6))
                plt.imshow(rgb_img)
                plt.title(img_name)
                plt.axis('off')
                plt.show()

        logger.info(f"Successfully saved {len(images)} annotated images.")