import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import urllib.request
from tqdm import tqdm

from helpers import DATA_DIR, PROJECT_ROOT, load_icebergs_by_frame
from segment_anything import SamPredictor, sam_model_registry

"""
Visualization Module

This module provides visualization functionality for iceberg tracking
data, including annotation rendering and video generation capabilities. It supports
multiple annotation sources (ground truth, detections, tracking results) and 
various visualization options for analysis and presentation purposes.

Main Components:
- Visualizer: Main orchestrator class for image annotation and video rendering
- SAM integration: Advanced segmentation using Segment Anything Model
- Multi-format annotation support: IDs, bounding boxes, contours, and masks
- Video generation utilities for temporal sequence visualization

Features:
- Flexible annotation rendering with customizable visual elements
- Consistent color mapping for object tracking across frames  
- Integration with Segment Anything Model for precise segmentation
- Progress tracking and batch processing capabilities
"""

class Visualizer:
    """
    A class for visualizing iceberg tracking data with various annotation options.

    This class provides functionality to annotate images with iceberg tracking information,
    including bounding boxes, IDs, contours, and masks. It can also render videos from
    the annotated image sequences.

    Attributes:
        dataset (str): Name of the dataset directory
        image_format (str): Image file format (e.g., '.JPG')
        annotation_source (str): Source of annotations ('gt', 'detections', or 'tracking')
        show_images (bool): Whether to display images during processing
        start_index (int): Starting frame index for processing
        length (int): Number of frames to process
        image_dir (str): Path to raw images directory
        device (str): Device for PyTorch operations ('cuda' or 'cpu')
        txt_file (str): Path to annotation file
        output_dir (str): Directory for saving annotated images
    """

    def __init__(self, dataset, image_format="JPG", annotation_source="tracking",
                 start_index=0, length=10, show_images=False):
        """
        Initialize the Visualizer with dataset configuration.

        Args:
            dataset (str): Name of the dataset directory
            image_format (str, optional): Image file format. Defaults to "JPG".
            annotation_source (str, optional): Source of annotations. Must be 'gt', 
                'detections', or 'tracking'. Defaults to "tracking".
            start_index (int, optional): Starting frame index. Defaults to 0.
            length (int, optional): Number of frames to process. Defaults to 10.
            show_images (bool, optional): Whether to display images. Defaults to False.

        Raises:
            ValueError: If annotation_source is not valid.
        """
        self.dataset = dataset
        self.image_format = f".{image_format}"
        self.annotation_source = annotation_source
        self.show_images = show_images
        self.start_index = start_index
        self.length = length

        # Set up directory paths
        self.image_dir = os.path.join(DATA_DIR, self.dataset, "images", "raw")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Configure annotation file and output directory based on source
        if self.annotation_source == "gt":
            self.txt_file = os.path.join(DATA_DIR, self.dataset, "annotations", "gt.txt")
            self.output_dir = os.path.join(DATA_DIR, self.dataset, "results", "images", "gt")
        elif self.annotation_source == "detections":
            self.txt_file = os.path.join(DATA_DIR, self.dataset, "detections", "det.txt")
            self.output_dir = os.path.join(DATA_DIR, self.dataset, "results", "images", "detections")
        elif self.annotation_source == "tracking":
            self.txt_file = os.path.join(DATA_DIR, self.dataset, "results", "mot.txt")
            self.output_dir = os.path.join(DATA_DIR, self.dataset, "results", "images", "tracking")
        else:
            raise ValueError(
                f"Invalid input type '{self.annotation_source}'. Must be either 'gt, 'detections' or 'tracking'")

    def annotate_icebergs(self, draw_ids=False, draw_boxes=False, draw_contours=False, draw_masks=False):
        """
        Annotate iceberg images with various visualization options.

        This method processes a sequence of images and adds annotations such as IDs,
        bounding boxes, contours, and segmentation masks for tracked icebergs.

        Args:
            draw_ids (bool, optional): Draw iceberg IDs. Defaults to False.
            draw_boxes (bool, optional): Draw bounding boxes. Defaults to False.
            draw_contours (bool, optional): Draw contours using SAM. Defaults to False.
            draw_masks (bool, optional): Draw segmentation masks using SAM. Defaults to False.
        """
        print("Start annotating icebergs as follows:")
        print(
            f"draw_ids = {draw_ids}, draw_boxes = {draw_boxes}, draw_contours = {draw_contours}, draw_masks = {draw_masks}")

        # Load iceberg data from annotation file
        icebergs_by_frame = load_icebergs_by_frame(self.txt_file)
        # Slice the data to process only the specified range
        icebergs_by_frame = dict(list(icebergs_by_frame.items())[self.start_index:self.start_index + self.length])

        # Initialize SAM predictor if contours or masks are needed
        if draw_contours or draw_masks:
            sam_predictor = self._get_sam_predictor()
        else:
            sam_predictor = None

        # Process all frames and generate annotated images
        images = self._map_icebergs(icebergs_by_frame, draw_ids, draw_boxes, draw_contours, draw_masks, sam_predictor)
        print("Finished annotating.")

        # Save the annotated images
        self._export_images(icebergs_by_frame, images)

    def render_video(self, fps=1):
        """
        Create a video from the annotated images.

        This method takes the annotated images and combines them into an MP4 video
        with the specified frame rate.

        Args:
            fps (int, optional): Frames per second for the output video. Defaults to 1.

        Raises:
            FileNotFoundError: If output directory doesn't exist or images are missing.
        """
        print(f"Start rendering video with {fps} fps...")

        # Set up video output directory and filename
        video_dir = os.path.join(DATA_DIR, self.dataset, "results", "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{self.annotation_source}_{self.start_index}-{self.start_index + self.length - 1}.mp4"
        video_path = os.path.join(video_dir, video_name)

        # Verify that annotated images exist
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"{self.output_dir} does not exist. Please run annotate_icebergs() first.")

        # Get list of input and output images
        images = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(self.image_format.lower())])
        images = images[self.start_index:self.start_index + self.length]
        output_images = [f for f in os.listdir(self.output_dir) if f.lower().endswith(self.image_format.lower())]

        # Check that all required annotated images exist
        for image in images:
            if image not in output_images:
                raise FileNotFoundError(f"Some output images are missing. Please run annotate_icebergs() first.")

        # Get video dimensions from first image
        first_image = cv2.imread(str(os.path.join(self.output_dir, images[0])))
        height, width, _ = first_image.shape

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write each frame to the video
        for frame_name in images:
            image_path = os.path.join(self.output_dir, frame_name)
            img = cv2.imread(image_path)
            video_writer.write(img)

        # Clean up and report completion
        video_writer.release()
        print("Finished rendering.")
        print(f"Video saved to {video_path}")

    def _get_sam_predictor(self):
        """
        Initialize and return a SAM (Segment Anything Model) predictor.

        Downloads the SAM model weights if they don't exist locally, then loads
        the model and returns a predictor instance.

        Returns:
            SamPredictor: Configured SAM predictor for segmentation tasks
        """
        print("Loading SAM model...")

        # Define SAM model paths and URLs
        sam_weights_filename = "sam_vit_b_01ec64.pth"
        sam_weights_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(sam_weights_dir, exist_ok=True)
        sam_weights_path = os.path.join(sam_weights_dir, sam_weights_filename)
        sam_weights_url = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_weights_filename

        # Download weights if they don't exist
        if not os.path.exists(sam_weights_path):
            print(f"Downloading SAM weights to: {sam_weights_path} (~375MB)")
            urllib.request.urlretrieve(sam_weights_url, sam_weights_path)
            print("Download complete.")

        # Load and configure the SAM model
        sam = sam_model_registry["vit_b"](checkpoint=sam_weights_path)
        sam.to(device=self.device)
        print("Loading complete.")

        return SamPredictor(sam)

    def _map_icebergs(self, icebergs_by_frame, draw_ids, draw_boxes, draw_contours, draw_masks, sam_predictor=None):
        """
        Process all frames and add iceberg annotations to images.

        This is the core processing method that iterates through all frames and
        applies the requested annotations to each iceberg in each frame.

        Args:
            icebergs_by_frame (dict): Dictionary mapping frame names to iceberg data
            draw_ids (bool): Whether to draw iceberg IDs
            draw_boxes (bool): Whether to draw bounding boxes
            draw_contours (bool): Whether to draw contours
            draw_masks (bool): Whether to draw segmentation masks
            sam_predictor (SamPredictor, optional): SAM predictor for segmentation

        Returns:
            list: List of annotated images as numpy arrays

        Raises:
            FileNotFoundError: If an image file cannot be found
        """
        images_with_mappings = []
        colormap = {}  # Store consistent colors for each iceberg ID

        # Calculate total number of entries for progress bar
        total_entries = sum(len(inner) for inner in icebergs_by_frame.values())

        # Initialize progress bar
        progress_bar = tqdm(
            total=total_entries,
            desc="Processing entries",
            unit="entry",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Process each frame
        for frame_name, icebergs in icebergs_by_frame.items():
            progress_bar.set_description(f"Annotating icebergs in {frame_name}{self.image_format}")

            # Load the image
            image_path = os.path.join(self.image_dir, frame_name + self.image_format)
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image file not found: {img}")

            # Prepare SAM predictor if needed for segmentation
            if draw_contours or draw_masks:
                outline_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sam_predictor.set_image(outline_img)

            # Process each iceberg in the current frame
            for iceberg_id, iceberg_data in icebergs.items():
                progress_bar.update(1)

                # Get consistent color for this iceberg ID
                color = self._get_object_color(iceberg_id, colormap)

                # Extract bounding box coordinates
                x1, y1, w, h = iceberg_data['bbox']
                x2 = x1 + w
                y2 = y1 + h

                # Draw iceberg ID if requested
                if draw_ids:
                    cv2.putText(img, str(iceberg_id), (int(x1), max(int(y1) - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw bounding box if requested
                if draw_boxes:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

                # Generate and draw segmentation if requested
                if draw_contours or draw_masks:
                    contours, mask, score = self._segment_icebergs(sam_predictor, np.array([x1, y1, x2, y2]))

                    # Apply mask overlay if requested
                    if draw_masks:
                        mask_alpha = 0.5  # Transparency level for mask overlay
                        img[mask] = ((1 - mask_alpha) * img[mask] + mask_alpha * np.array(color)).astype(np.uint8)

                    # Draw contours if requested
                    if draw_contours:
                        cv2.drawContours(img, contours, -1, color, 2)

            images_with_mappings.append(img)

        progress_bar.close()
        return images_with_mappings

    def _get_object_color(self, object_id, colormap):
        """
        Get a consistent color for an object ID.

        Uses a seeded random number generator to ensure the same object ID
        always gets the same color across frames.

        Args:
            object_id (int): Unique identifier for the object
            colormap (dict): Dictionary storing ID to color mappings

        Returns:
            tuple: RGB color tuple (B, G, R) for OpenCV
        """
        if object_id not in colormap:
            # Use object ID as seed for consistent color generation
            np.random.seed(object_id)
            color = tuple(map(int, np.random.randint(0, 255, size=3)))
            colormap[object_id] = color

        return colormap[object_id]

    def _segment_icebergs(self, sam_predictor, input_box):
        """
        Generate segmentation mask and contours for an iceberg using SAM.

        Uses the Segment Anything Model to generate a segmentation mask for
        the iceberg within the given bounding box, then extracts contours.

        Args:
            sam_predictor (SamPredictor): Configured SAM predictor
            input_box (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            tuple: (contours, mask, score) where:
                - contours: OpenCV contours for drawing
                - mask: Boolean mask array
                - score: Confidence score from SAM
        """
        # Run SAM prediction using bounding box as prompt
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # Add batch dimension
            multimask_output=False
        )

        # Select the best mask based on confidence score
        valid_masks = scores >= 0
        best_idx = np.argmax(scores[valid_masks])
        mask = masks[valid_masks][best_idx]
        score = scores[valid_masks][best_idx]

        # Convert mask to uint8 format for contour extraction
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour if multiple are found
        if len(contours) > 1:
            contours = contours[0] if contours[0].shape[0] > contours[1].shape[0] else contours[1]

        return contours, mask, score

    def _export_images(self, icebergs_by_frame, images):
        """
        Save annotated images to the output directory.

        Saves all processed images to the configured output directory and
        optionally displays them using matplotlib.

        Args:
            icebergs_by_frame (dict): Dictionary mapping frame names to iceberg data
            images (list): List of annotated images as numpy arrays
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Save each annotated image
        for index, frame_name in enumerate(icebergs_by_frame):
            img = images[index]

            # Convert BGR to RGB for display purposes
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save the image (OpenCV uses BGR format)
            img_name = f"{frame_name}{self.image_format}"
            output_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(output_path, img)

            # Display image if requested
            if self.show_images:
                plt.figure(figsize=(10, 6))
                plt.imshow(rgb_img)
                plt.title(img_name)
                plt.axis('off')
                plt.show()

        print(f"Saved images in {self.output_dir}.")


def main():
    # Configuration parameters
    dataset = "hill_2min_2023-08"
    image_format = "JPG"

    # Initialize visualizer with tracking data
    visualizer = Visualizer(dataset, image_format, annotation_source="tracking", start_index=0, length=10, show_images=True)

    # Generate annotated images with all visualization options enabled
    visualizer.annotate_icebergs(draw_ids=True, draw_boxes=True, draw_contours=True, draw_masks=True)

    # Create a video from the annotated images
    visualizer.render_video(fps=1)


if __name__ == '__main__':
    main()