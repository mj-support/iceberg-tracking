import os
import cv2
import numpy as np
import pandas as pd
from utils.helpers import DATA_DIR
from IPython.display import HTML
import base64
import matplotlib.pyplot as plt


class Visualizer:
    """
    A class for visualizing iceberg detection and tracking results.

    This class provides functionality to visualize different stages of an iceberg detection
    and tracking pipeline. It can display raw images, preprocessed images, detection results,
    or tracking results. The visualization can be displayed as individual plots or compiled
    into a video.

    Attributes:
        dataset (str): Name of the dataset to visualize
        image_format (str): Format of the image files (jpg, png, etc.)
        stage (str): Processing stage to visualize (preprocessing, detection, tracking)
        segment (bool): If True, visualize segmentation
        start_index (int): Starting index for image selection
        length (int): Number of images to process
        image_dir (str): Directory containing the images
        images (list): List of image paths
    """

    def __init__(self, dataset, image_format="jpg", stage="tracking", image_dir="raw", segment=False, start_index=0, length=10):
        """
        Initialize the Visualizer with dataset and visualization parameters.

        Args:
            dataset (str): Name of the dataset to visualize
            image_format (str): Format of image files without dot (e.g., 'jpg', 'png')
            stage (str): Processing stage to visualize. Options:
                         - 'preprocessing': Visualize preprocessed images
                         - 'detection': Visualize detection results
                         - 'tracking': Visualize tracking results
            image_dir (str): Directory containing the images. Options:
                            - 'raw': Visualize raw images
                            - 'preprocessed': Visualize preprocessed images
            segment (bool): If True, visualize segmentation, else bounding boxes
            start_index (int): Starting index for image selection
            length (int): Number of images to process

        Raises:
            ValueError: If an invalid stage is provided
        """
        # Validate the stage parameter
        valid_stages = {"preprocessing", "detection", "tracking"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")
        valid_image_dir = {"raw", "processed"}
        if image_dir not in valid_image_dir:
            raise ValueError(f"Invalid image directory '{image_dir}'. Must be one of {valid_image_dir}")

        # Store initialization parameters
        self.dataset = dataset
        self.image_format = f".{image_format}"
        self.stage = stage
        self.segment = segment
        self.start_index = start_index
        self.length = length
        self.image_dir = os.path.join(DATA_DIR, dataset, "images", image_dir)

        if stage == "detection":
            # For detection stage, we need the detection results file
            self.txt_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
        if stage == "tracking":
            # For tracking stage, we need the tracking results file
            self.txt_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

        # Get a sorted list of image files with the specified format
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.image_format.lower())
        ])

        # Trim the image list to the desired range
        self.images = self.images[start_index:start_index + length]

    def pyplot(self, save_images=False):
        """
        Display all processed images individually using matplotlib.

        This method shows each image with its bounding boxes and object IDs
        in a separate figure using matplotlib. Useful for detailed inspection
        of individual frames.

        Args:
            save_images (bool): If True, saves each image to disk instead of just displaying it.
        """
        images = []
        if self.stage == "preprocessing":
            for i, img_name in enumerate(self.images):
                # Read the image from disk
                image_path = os.path.join(self.image_dir, img_name)
                img = cv2.imread(image_path)
                images.append(img)
        else:
            # Get images with bounding boxes and object IDs mapped
            images = self._map_icebergs()

        # Optional: create output directory
        if save_images:
            output_dir = os.path.join(DATA_DIR, self.dataset, "results", "images")
            os.makedirs(output_dir, exist_ok=True)

        # Display or save each image
        for index, img in enumerate(images):
            # Convert from BGR (OpenCV format) to RGB (matplotlib format)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if save_images:
                img_name = f"{self.images[index]}"
                output_path = os.path.join(output_dir, img_name)
                cv2.imwrite(output_path, img)
            else:
                plt.figure(figsize=(10, 6))
                plt.imshow(rgb_img)
                plt.title(self.images[index])
                plt.axis('off')
                plt.show()

    def render_video(self, fps=1, resolution=None):
        """
        Render a video from the visualization results.

        Creates a video file from the selected images with bounding boxes and object IDs
        when applicable (detection and tracking stages). The video is saved to the dataset's
        video directory.

        Args:
            fps (int): Frames per second for the output video
            resolution (tuple, optional): Desired resolution as (width, height).
                                         If None, uses the original image resolution.

        Returns:
            HTML: HTML object for displaying the video in Jupyter notebooks
        """
        # Create video directory if it doesn't exist
        video_dir = os.path.join(DATA_DIR, self.dataset, "videos/")
        os.makedirs(os.path.dirname(video_dir), exist_ok=True)
        video_path = os.path.join(video_dir, f"{self.stage}.mp4")
        if self.segment:
            video_path = os.path.join(video_dir, f"{self.stage}_segment.mp4")

        # Set video dimensions based on resolution parameter or first image
        if resolution:
            width, height = resolution
        else:
            first_image = cv2.imread(str(os.path.join(self.image_dir, self.images[0])))
            height, width, _ = first_image.shape

        # Initialize the video writer with MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 output
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        images = []
        # Get images with bounding boxes and object IDs mapped
        images = self._map_icebergs()

        # Process each image and write to video
        for img in images:
            if resolution:
                img = cv2.resize(img, (width, height))

            # Write the processed image to video
            video_writer.write(img)

        # Release resources and display the video
        video_writer.release()
        print(f"Video saved to {video_path}")
        return self._display_video(video_path)

    def _map_icebergs(self):
        """
        Map icebergs by drawing bounding boxes and IDs on images.

        Reads detection or tracking data from the corresponding text file and
        draws bounding boxes around detected/tracked objects. Each object ID gets
        a consistent color for easier tracking visualization.

        Args:
            segment (bool): If True, perform iceberg segmentation. If False, only draw bounding boxes.

        Returns:
            list: List of images with bounding boxes and object IDs drawn on them
        """
        # Load detection/tracking data
        det_data = pd.read_csv(self.txt_file, header=None)
        # Assign column names based on expected format
        det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                            'confidence', 'x', 'y', 'z']

        # Initialize color mapping dictionary for consistent object coloring
        colormap = {}

        images_with_mappings = []

        # Process each image in the selected range
        for img_name in self.images:
            # Load image and get frame ID
            image_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Image {image_path} not found.")
                continue

            # Extract frame ID from image name (removing file extension)
            image_no = img_name[:-4]

            # Process the image based on whether segmentation is enabled
            if self.segment:
                processed_img = self._process_image_with_segmentation(img, det_data, image_no, colormap)
            else:
                processed_img = self._process_image_with_bounding_boxes(img, det_data, image_no, colormap)

            images_with_mappings.append(processed_img)

        return images_with_mappings

    def _display_video(self, video_path, width=640, height=480):
        """
        Display a video inside a Jupyter Notebook.

        Encodes the video as base64 and creates an HTML video element to display it.

        Args:
            video_path (str): Path to the video file
            width (int): Display width of the video
            height (int): Display height of the video

        Returns:
            HTML: HTML object containing the video for Jupyter notebook display
        """
        # Read the video file and encode it as base64
        video = open(video_path, "rb").read()
        encoded = base64.b64encode(video).decode("ascii")

        # Create HTML with centered video element
        return HTML(f"""
        <div style="text-align: center;">
            <video width="{width}" height="{height}" controls>
                <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
            </video>
        </div>
        """)

    def _segment_largest_iceberg_in_box(self, img, box, padding=0):
        """
        Segment only the largest iceberg within a specified bounding box using color-based segmentation.

        Args:
            img (ndarray): The input image
            box (tuple): (x0, y0, width, height) - The bounding box coordinates
            padding (int): Optional padding to add around the bounding box

        Returns:
            tuple: (cropped image, mask, segmented image)
        """
        # Extract box coordinates with padding
        x0, y0, width, height = box
        x0 = max(0, int(x0 - padding))
        y0 = max(0, int(y0 - padding))
        x1 = min(img.shape[1], int(x0 + width + 2 * padding))
        y1 = min(img.shape[0], int(y0 + height + 2 * padding))

        # Crop the image to the bounding box
        cropped = img[y0:y1, x0:x1]

        # Convert cropped image to HSV and LAB color spaces
        cropped_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        cropped_lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)

        # Extract channels
        h, s, v = cv2.split(cropped_hsv)
        l, a, b = cv2.split(cropped_lab)

        # Create masks for potential iceberg regions
        # Icebergs are white/gray (high value, low saturation in HSV)
        mask_value = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)[1]  # High brightness
        mask_sat = cv2.threshold(s, 70, 255, cv2.THRESH_BINARY_INV)[1]  # Low saturation

        # Combine masks
        mask_combined = cv2.bitwise_and(mask_value, mask_sat)

        # Additional LAB color space thresholding
        # White/gray areas have high L values
        mask_l = cv2.threshold(l, 160, 255, cv2.THRESH_BINARY)[1]

        # Combine with previous mask
        mask_combined = cv2.bitwise_and(mask_combined, mask_l)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask_morph = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create final mask with only the largest contour
        final_mask = np.zeros_like(mask_morph)

        largest_contour = None
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Only use contours that are large enough
            min_contour_area = 50  # Lower threshold for smaller bounding boxes
            if cv2.contourArea(largest_contour) > min_contour_area:
                cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)

        # Apply the mask to the cropped image
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        segmented = cv2.bitwise_and(cropped_rgb, cropped_rgb, mask=final_mask)

        return cropped_rgb, final_mask, segmented, largest_contour

    def _process_image_with_bounding_boxes(self, img, det_data, image_no, colormap):
        """
        Process an image by drawing bounding boxes and object IDs.

        This method adds visual identification of icebergs by drawing rectangular
        bounding boxes around each detected object and labeling them with their ID.
        Each object ID is consistently assigned the same color.

        Args:
            img (numpy.ndarray): The original image to draw on
            det_data (pandas.DataFrame): Detection data containing bounding box coordinates
            image_no (str): Current frame/image ID to filter detections
            colormap (dict): Dictionary mapping object IDs to color tuples

        Returns:
            numpy.ndarray: Image with bounding boxes and IDs drawn on it
        """
        # Filter detection/tracking data for the current frame
        frame_data = det_data[det_data['frame'] == image_no]

        # Draw bounding boxes and labels for each object in the frame
        for _, row in frame_data.iterrows():
            object_id = int(row['ID'])
            color = self._get_object_color(object_id, colormap)

            # Extract bounding box coordinates
            bbox_left = int(row['bbox_left'])
            bbox_top = int(row['bbox_top'])
            bbox_width = int(row['bbox_width'])
            bbox_height = int(row['bbox_height'])

            if row['confidence'] > 0.0:
                # Draw the bounding box around the iceberg
                cv2.rectangle(img, (bbox_left, bbox_top),
                              (bbox_left + bbox_width, bbox_top + bbox_height), color, 2)

                # Add the object ID label above the bounding box
                cv2.putText(img, str(object_id), (bbox_left, bbox_top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

    def _process_image_with_segmentation(self, img, det_data, image_no, colormap):
        """
        Process an image using segmentation for each detected iceberg object.

        This method performs more advanced visualization by:
        1. Finding the actual contours of each iceberg within its bounding box
        2. Filling each contour with a semi-transparent color unique to that object
        3. Labeling each segmented iceberg with its ID

        Unlike the simple bounding box approach, this creates a more precise
        visualization of the actual iceberg shapes.

        Args:
            img (numpy.ndarray): The original image to draw on
            det_data (pandas.DataFrame): Detection data containing bounding box coordinates
            image_no (str): Current frame/image ID to filter detections
            colormap (dict): Dictionary mapping object IDs to color tuples

        Returns:
            numpy.ndarray: Image with colored segmentation and IDs drawn on it
        """
        # Create separate image copies for contours and colored overlay
        img_contours = img.copy()  # Will contain the original image with contour lines and labels
        overlay = img.copy()  # Will contain colored regions for blending
        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # Track segmented areas

        # Filter detection/tracking data for the current frame
        frame_data = det_data[det_data['frame'] == image_no]

        # Process each detected object
        for _, row in frame_data.iterrows():
            object_id = int(row['ID'])
            color = self._get_object_color(object_id, colormap)

            # Get bounding box with padding for segmentation
            padding = 10  # Extra space around the bounding box for better segmentation
            bbox_left = int(row['bbox_left'])
            bbox_top = int(row['bbox_top'])
            bbox_width = int(row['bbox_width'])
            bbox_height = int(row['bbox_height'])

            # Apply padding while ensuring the box stays within image bounds
            x0_pad = max(0, bbox_left - padding)
            y0_pad = max(0, bbox_top - padding)
            width_pad = min(img.shape[1] - x0_pad, bbox_width + 2 * padding)
            height_pad = min(img.shape[0] - y0_pad, bbox_height + 2 * padding)
            box = (x0_pad, y0_pad, width_pad, height_pad)

            # Perform segmentation to find the iceberg contour
            _, mask, _, contour = self._segment_largest_iceberg_in_box(img, box)

            if contour is not None:
                # Adjust contour coordinates to the full image space
                adjusted_contour = contour.copy()
                adjusted_contour[:, :, 0] += x0_pad  # Shift x coordinates
                adjusted_contour[:, :, 1] += y0_pad  # Shift y coordinates

                # Create a temporary mask for this specific iceberg
                temp_mask = np.zeros_like(full_mask)
                cv2.drawContours(temp_mask, [adjusted_contour], -1, 255, -1)

                # Fill the contour with color in the overlay image
                cv2.drawContours(overlay, [adjusted_contour], -1, color, -1)

                # Calculate position for the ID label based on contour bounds
                x, y, w, h = cv2.boundingRect(adjusted_contour)
                label_x = x + w // 2  # Center horizontally within bounding rectangle
                label_y = max(y - 10, 10)  # Position above rectangle or adjust if out of bounds

                # Draw object ID on both images (contours and overlay)
                cv2.putText(img_contours, f"{object_id}", (label_x - 30, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(overlay, f"{object_id}", (label_x - 30, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Blend the colored overlay with the original image for semi-transparent effect
        alpha = 0.5  # Transparency level (0.5 = 50% transparent)
        colored_transparent = cv2.addWeighted(overlay, alpha, img_contours, 1 - alpha, 0)

        return colored_transparent

    def _get_object_color(self, object_id, colormap):
        """
        Get a consistent color for an object ID.

        This method ensures that each unique object ID always gets the same color
        across all frames, making it easier to visually track specific icebergs.
        The color is generated deterministically based on the object ID, and
        cached in the colormap dictionary for reuse.

        Args:
            object_id (int): The ID of the detected object
            colormap (dict): Dictionary mapping object IDs to color tuples

        Returns:
            tuple: RGB color tuple for the object (in BGR format for OpenCV)
        """
        # Check if we've already assigned a color to this object ID
        if object_id not in colormap:
            # Set random seed based on ID for deterministic color generation
            np.random.seed(object_id)  # Ensure consistent color assignment across runs

            # Generate a random RGB color (BGR for OpenCV) using values from 0-255
            color = tuple(map(int, np.random.randint(0, 255, size=3)))

            # Cache the color in the colormap
            colormap[object_id] = color

        return colormap[object_id]


def main():
    # Specify dataset and visualization parameters
    dataset = "hill_2min_2023-08"
    # Create visualizer object with tracking visualization
    #visualizer = Visualizer(dataset, image_format="JPG", stage="tracking", segment=False, start_index=0, length=10)
    visualizer = Visualizer(dataset, image_format="JPG", stage="tracking", image_dir="raw", segment=True, start_index=0, length=10)
    visualizer.txt_file = "/data/hill_2min_2023-08/results/mot.txt"
    #visualizer.txt_file = "/Users/marco/repos/iceberg-tracking/data/hill_2min_2023-08/detections/det.txt"

    # Display individual frames
    visualizer.pyplot(save_images=True)
    # Render and display video
    #visualizer.render_video()


if __name__ == "__main__":
    main()