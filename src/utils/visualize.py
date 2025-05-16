import os
import cv2
import numpy as np
import pandas as pd
from utils.paths import DATA_DIR
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
        start_index (int): Starting index for image selection
        length (int): Number of images to process
        image_dir (str): Directory containing the images
        images (list): List of image filenames to process
    """

    def __init__(self, dataset, image_format="jpg", stage="tracking", start_index=0, length=10):
        """
        Initialize the Visualizer with dataset and visualization parameters.

        Args:
            dataset (str): Name of the dataset to visualize
            image_format (str): Format of image files without dot (e.g., 'jpg', 'png')
            stage (str): Processing stage to visualize. Options:
                         - 'preprocessing': Visualize preprocessed images
                         - 'detection': Visualize detection results
                         - 'tracking': Visualize tracking results
            start_index (int): Starting index for image selection
            length (int): Number of images to process

        Raises:
            ValueError: If an invalid stage is provided
        """
        # Validate the stage parameter
        valid_stages = {"preprocessing", "detection", "tracking"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

        # Store initialization parameters
        self.dataset = dataset
        self.image_format = f".{image_format}".lower()
        self.stage = stage
        self.start_index = start_index
        self.length = length

        # Set up the appropriate image directory based on the stage
        self.image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
        if stage == "preprocessing":
            self.image_dir = os.path.join(DATA_DIR, dataset, "images", "processed")
        elif stage == "detection":
            # For detection stage, we need the detection results file
            self.txt_file = os.path.join(DATA_DIR, dataset, "detections", "det.txt")
        elif stage == "tracking":
            # For tracking stage, we need the tracking results file
            self.txt_file = os.path.join(DATA_DIR, dataset, "results", "mot.txt")

        # Get a sorted list of image files with the specified format
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.image_format)
        ])

        # Trim the image list to the desired range
        self.images = self.images[start_index:start_index + length]

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
        if self.stage == "preprocessing":
            for i, img_name in enumerate(self.images):
                # Read the image from disk
                image_path = os.path.join(self.image_dir, img_name)
                img = cv2.imread(image_path)
                images.append(img)
        else:
            # Get images with bounding boxes and object IDs mapped
            images = self.map_icebergs()

        # Process each image and write to video
        for img in images:
            if resolution:
                img = cv2.resize(img, (width, height))

            # Write the processed image to video
            video_writer.write(img)

        # Release resources and display the video
        video_writer.release()
        print(f"Video saved to {video_path}")
        return self.display_video(video_path)

    def map_icebergs(self):
        """
        Map icebergs by drawing bounding boxes and IDs on images.

        Reads detection or tracking data from the corresponding text file and
        draws bounding boxes around detected/tracked objects. Each object ID gets
        a consistent color for easier tracking visualization.

        Returns:
            list: List of images with bounding boxes and object IDs drawn on them
        """
        colormap = {}  # Dictionary to store colors for each object ID
        # Read detection/tracking data
        det_data = pd.read_csv(self.txt_file, header=None)
        # Assign column names based on expected format
        det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'x', 'y',
                            'z']

        images_with_mappings = []

        # Process each image in the selected range
        for i, img_name in enumerate(self.images):
            # Read the image from disk
            image_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(image_path)

            # Extract frame ID from image name (removing file extension)
            image_no = img_name[:-4]  # Get the frame ID from image name

            if img is None:
                print(f"Image {image_path} not found.")
                continue

            # Filter detection/tracking data for the current frame
            frame_data = det_data[det_data['frame'] == image_no]

            # Draw bounding boxes and labels for each object in the frame
            for _, row in frame_data.iterrows():
                object_id = int(row['ID'])

                # Assign a consistent color to each object ID
                if object_id not in colormap:
                    np.random.seed(object_id)  # Ensure consistent color assignment
                    color = tuple(map(int, np.random.randint(0, 255, size=3)))
                    colormap[object_id] = color
                color = colormap[object_id]

                # Extract bounding box coordinates from the data
                bbox_left, bbox_top = int(row['bbox_left']), int(row['bbox_top'])
                bbox_width, bbox_height = int(row['bbox_width']), int(row['bbox_height'])

                # Draw the bounding box and object ID label on the image
                cv2.rectangle(img, (bbox_left, bbox_top), (bbox_left + bbox_width, bbox_top + bbox_height), color, 2)
                cv2.putText(img, str(object_id), (bbox_left, bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Add the processed image to the result list
            images_with_mappings.append(img)

        return images_with_mappings

    def display_video(self, video_path, width=640, height=480):
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

    def pyplot(self):
        """
        Display all processed images individually using matplotlib.

        This method shows each image with its bounding boxes and object IDs
        in a separate figure using matplotlib. Useful for detailed inspection
        of individual frames.
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
            images= self.map_icebergs()

        # Display each image in a separate matplotlib figure
        for index, img in enumerate(images):
            # Create a new figure for each image
            plt.figure(figsize=(10, 6))
            # Convert from BGR (OpenCV format) to RGB (matplotlib format)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(self.images[index])
            plt.axis('off')
            plt.show()


def main():
    # Specify dataset and visualization parameters
    dataset = "fjord_2min_2023-08"
    # Create visualizer object with tracking visualization
    visualizer = Visualizer(dataset, image_format="JPG", stage="preprocessing", start_index=0, length=10)
    # Display individual frames
    visualizer.pyplot()
    # Render and display video
    visualizer.render_video()


if __name__ == "__main__":
    main()