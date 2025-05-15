import os
from PIL import Image, ImageEnhance, ImageStat
from PIL.ExifTags import TAGS
from datetime import datetime
import numpy as np
import cv2
from skimage.io import imsave
from utils.paths import DATA_DIR
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class ImagePreprocessor:
    """
    A class for preprocessing images in a dataset with various operations.

    This preprocessor can perform three main operations:
    1. Brightness adjustment for nighttime images
    2. Masking specific areas (green areas by default)
    3. Tiling (splitting images into predefined regions)

    All operations are optional and can be enabled/disabled at initialization.
    """

    def __init__(self, dataset, image_format, brighten=False, mask=False, tile=False, night_start=None, night_end=None):
        """
        Initialize the image preprocessor with specified operations as parameters.

        Args:
            dataset (str): Name of the dataset stored in data/
            image_format (str): File formats of the images (file extension)
            brighten (bool): Whether to apply brightness enhancement
            mask (bool): Whether to apply masking
            tile (bool): Whether to split images into tiles
            night_start (int): Hour (0-23) marking start of night for brightness enhancement
            night_end (int): Hour (0-23) marking end of night for brightness enhancement

        Raises:
            ValueError: If brighten is True but night_start or night_end is not provided
            ValueError: If night_start or night_end is not between 0 and 23
            FileNotFoundError: If the input directory does not exist
        """
        self.dataset = dataset
        self.image_format = f".{image_format}"
        self.brighten = brighten
        self.mask = mask
        self.tile = tile

        # Validate night_start and night_end if brighten is enabled
        if self.brighten:
            if night_start is None or night_end is None:
                raise ValueError("night_start and night_end must be specified when brighten=True")

            if not (0 <= night_start <= 23) or not (0 <= night_end <= 23):
                raise ValueError("night_start and night_end must be between 0 and 23")

            self.night_start = night_start
            self.night_end = night_end

        # Set up input and output directories
        self.input_dir = os.path.join(DATA_DIR, self.dataset, "images", "raw")
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"❗ Please store all raw and unprocessed images in: {self.input_dir}")

        self.output_dir = os.path.join(DATA_DIR, self.dataset, "images", "processed")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Keep track of preprocessing operations for logging
        self.preprocessing_ops = []
        if self.brighten:
            self.preprocessing_ops.append(f"brighten (night hours: {night_start}-{night_end})")
        if self.mask:
            self.preprocessing_ops.append("mask")
        if self.tile:
            self.preprocessing_ops.append("tile")

        print(f"ImagePreprocessor initialized with operations: {', '.join(self.preprocessing_ops)}")
        print(f"The preprocessed images will be saved at: {self.output_dir}")

    def process_images(self):
        """
        Process all images in the input directory with the enabled preprocessing operations.

        This method:
        1. Iterates through all images in the input directory
        2. Applies brightness adjustment if enabled and if the image was taken at night
        3. Applies masking if enabled
        4. Tiles the image if enabled, otherwise saves the processed image directly

        The processing is applied sequentially: brighten -> mask -> tile.
        Displays a progress bar to show processing status.
        """
        # Get list of image files
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(self.image_format)]

        # Cache the green mask if masking is enabled (to avoid recomputing for each image)
        green_mask = self._extract_green_mask() if self.mask else None
        # Cache the tiles configuration if tiling is enabled
        tiles = get_tiles_with_overlap(self.dataset) if self.tile else None

        # Create a progress bar using tqdm
        progress_bar = tqdm(
            image_files,
            desc="Processing images",
            unit="image",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Process each image with progress tracking
        for image_name in progress_bar:
            image_path = os.path.join(self.input_dir, image_name)
            image = Image.open(image_path)

            # Update progress bar description with current file
            progress_bar.set_description(f"Processing {image_name}")

            # Apply brightness adjustment if enabled and if image was taken at night
            if self.brighten:
                time_of_day = self._get_image_time_of_day(image_path)
                if time_of_day == "night":  # Adjust the brightness
                    image = self._adjust_brightness(image, target_brightness=70, tolerance=2)

            # Apply masking if enabled
            if self.mask:
                image = self._apply_mask(image, green_mask)

            # Apply tiling if enabled, otherwise save the processed image directly
            if self.tile:
                self._tile_and_save_images(image, image_name, tiles)
            else:
                imsave(os.path.join(self.output_dir, image_name), image)

    def _get_image_time_of_day(self, image_path):
        """
        Determine if an image was taken during day or night based on EXIF data.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: "night" if the image was taken between night_start and night_end hours,
                 "day" if taken during daytime, or None if no timestamp was found
        """
        image = Image.open(image_path)

        # Get EXIF data from the image
        exif_data = image._getexif()
        if not exif_data:
            return None

        # Look for "DateTimeOriginal"-Tag in the EXIF data
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "DateTimeOriginal":
                image_datetime = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

                # Check if it is night time based on the configured night hours
                if self.night_start <= image_datetime.hour < self.night_end:
                    return "night"
                else:
                    return "day"

        return None

    def _calculate_brightness(self, image):
        """
        Calculate the average brightness of an image.

        Args:
            image (PIL.Image): The input image

        Returns:
            float: Average brightness value (0-255)
        """
        # Convert image to grayscale
        grayscale_image = image.convert("L")
        # Calculate the mean brightness
        stat = ImageStat.Stat(grayscale_image)
        brightness = stat.mean[0]
        return brightness

    def _adjust_brightness(self, image, target_brightness=70, tolerance=2):
        """
        Adjust the brightness of an image to reach a target brightness.

        Args:
            image (PIL.Image): The input image
            target_brightness (float): The desired brightness level (0-255)
            tolerance (float): Acceptable difference from target brightness

        Returns:
            PIL.Image: Image with adjusted brightness
        """
        current_brightness = self._calculate_brightness(image)
        iteration = 0

        # Iteratively adjust brightness until within tolerance of target
        while abs(current_brightness - target_brightness) > tolerance:
            iteration += 1
            # Calculate factor needed to reach target brightness
            brightness_factor = target_brightness / current_brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
            current_brightness = self._calculate_brightness(image)

        return image

    def _extract_green_mask(self):
        """
        Extract a boolean mask identifying green areas from a reference image.

        Looks for a file named 'mask.jpg' in the dataset directory and extracts
        green areas using HSV color thresholding.

        Returns:
            numpy.ndarray: Boolean mask where True indicates green pixels

        Raises:
            FileNotFoundError: If mask.jpg is not found in the dataset directory
        """
        image_path = os.path.join(DATA_DIR, self.dataset, "images", "mask.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❗ Please save the masking image as follows: {image_path}")

        image = Image.open(image_path)
        image_array = np.array(image)

        # Convert the image from RGB to HSV for better color separation
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

        # Define the green color range in HSV color space
        # H: 35-85 (green hues), S: 100-255 (moderately to fully saturated), V: 100-255 (moderately bright to bright)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # Create a mask for the green area in the specified range
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        green_mask = green_mask.astype(bool)
        return green_mask

    def _apply_mask(self, image, mask):
        """
        Apply a binary mask to an image, setting masked areas to black.

        Args:
            image (PIL.Image): The input image
            mask (numpy.ndarray): Boolean mask where True indicates pixels to be masked

        Returns:
            PIL.Image: The masked image
        """
        image_array = np.array(image)

        # Copy the original image array
        masked_image = image_array.copy()

        # Set the masked area (where mask is True) to black
        masked_image[mask] = [0, 0, 0]  # This sets the masked area to black
        masked_image = Image.fromarray(masked_image)
        return masked_image

    def _tile_and_save_images(self, image, image_name, tiles):
        """
        Extract tiles from an image based on predefined coordinates and save them.

        Args:
            image (PIL.Image): The input image to tile
            image_name (str): Original filename of the image
            tiles (dict): Dictionary of tile coordinates
        """
        for tile_entry in tiles:
            tile = tiles[tile_entry]
            # Create output filename with tile identifier
            output_img_path = os.path.join(self.output_dir, image_name[:-4] + '_' + tile_entry + self.image_format)

            # Convert PIL image to numpy array for slicing
            image_array = np.array(image)

            # Extract the tile using coordinate slicing
            output_img_tile = image_array[tile["ymin"]:tile["ymax"], tile["xmin"]:tile["xmax"]]

            # Save the tile to the output directory
            imsave(output_img_path, output_img_tile)

    def _tiling_helper(self):
        """
        Visualization tool to help with defining tile coordinates.

        This method:
        1. Selects a random image from the processed directory
        2. Displays it with a grid overlay at 1000-pixel intervals
        3. Helps visually determine appropriate tile coordinates

        This is a development/debugging tool and not used in normal processing.
        """
        # Directory where your images are located
        image_dir = os.path.join(DATA_DIR, self.dataset, "images", "processed")
        # List all the files in the directory
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        # Select a random image file from the list
        random_image_file = random.choice(image_files)
        # Construct the full path to the randomly selected image
        image_path = os.path.join(image_dir, random_image_file)

        # Open the image
        image = Image.open(image_path)

        # Get image dimensions
        width, height = image.size

        # Create visualization with grid lines
        fig, ax = plt.subplots()
        ax.imshow(image)

        # Set grid lines at every 1000-pixel value
        tick_spacing = 1000

        # Draw horizontal grid lines
        for y in range(0, height, tick_spacing):
            ax.axhline(y, color='red', linestyle='--', linewidth=0.8)

        # Draw vertical grid lines
        for x in range(0, width, tick_spacing):
            ax.axvline(x, color='blue', linestyle='--', linewidth=0.8)

        plt.show()


def get_tiles_with_overlap(dataset):
    """
    Define tile coordinates for different dataset types with overlap between tiles.

    The overlap is calculated as a percentage (default 2.5%) of the tile dimensions
    to ensure smooth transitions between tiles.

    Returns:
        dict: Dictionary of tile coordinates with keys for each tile region

    Raises:
        NotImplementedError: If the dataset is not recognized (currently must start with 'hill' or 'fjord')
    """
    overlap = 0.025  # 2.5% overlap between tiles
    # self._tiling_helper()  # Uncomment to visualize tiling grid

    # Define tile coordinates based on dataset type
    if dataset.startswith("hill"):
        # Hill dataset tile definitions
        tiles = {
            "A": {"xmin": 0, "xmax": 2000, "ymin": 1000, "ymax": 2300},
            "B": {"xmin": 2000, "xmax": 4000, "ymin": 1000, "ymax": 2300},
            "C": {"xmin": 4000, "xmax": 6000, "ymin": 1400, "ymax": 2900},
            "D": {"xmin": 0, "xmax": 2000, "ymin": 2300, "ymax": 3600},
            "E": {"xmin": 2000, "xmax": 4000, "ymin": 2300, "ymax": 3600},
        }

        # Adjust tile boundaries to create overlap
        tiles["B"]["xmin"] = int(tiles["B"]["xmin"] * (1 - overlap))
        tiles["C"]["xmin"] = int(tiles["C"]["xmin"] * (1 - overlap))
        tiles["D"]["ymin"] = int(tiles["D"]["ymin"] * (1 - overlap))
        tiles["E"]["ymin"] = int(tiles["E"]["ymin"] * (1 - overlap))
        tiles["E"]["xmin"] = int(tiles["E"]["xmin"] * (1 - overlap))

    elif dataset.startswith("fjord"):
        # Fjord dataset tile definitions
        tiles = {
            "A": {"xmin": 0, "xmax": 1000, "ymin": 1000, "ymax": 2250},
            "B": {"xmin": 1000, "xmax": 2000, "ymin": 500, "ymax": 2250},
            "C": {"xmin": 2000, "xmax": 3000, "ymin": 500, "ymax": 2000},
            "D": {"xmin": 3000, "xmax": 4000, "ymin": 500, "ymax": 1750},
            "E": {"xmin": 4000, "xmax": 5750, "ymin": 750, "ymax": 1500},
        }

        # Adjust tile boundaries to create overlap
        tiles["B"]["xmin"] = int(tiles["B"]["xmin"] * (1 - overlap))
        tiles["C"]["xmin"] = int(tiles["C"]["xmin"] * (1 - overlap))
        tiles["D"]["xmin"] = int(tiles["D"]["xmin"] * (1 - overlap))
        tiles["E"]["xmin"] = int(tiles["E"]["xmin"] * (1 - overlap))

    else:
        raise NotImplementedError(f"❗ Please implement the tile sizes for this dataset: {dataset}\n"
                                  f"If the dataset is from the fjord or hill, it should be name after it")

    return tiles


def main():
    dataset = "hill_2min_2023-08"

    # Creates an ImagePreprocessor instance with the specified dataset and preprocessing options
    preprocessor = ImagePreprocessor(
        dataset=dataset,
        image_format="JPG",
        brighten=True,
        mask=True,
        tile=True,
        night_start=0,  # 0 AM
        night_end=6  # 6 AM
    )

    # Run the preprocessing pipeline
    preprocessor.process_images()

    from utils.visualize import visualize
    visualize(dataset, None, start_index=200, length=10, save_images=False)




if __name__ == "__main__":
    main()