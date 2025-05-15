import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.paths import DATA_DIR
from ipywidgets import interact, IntSlider, VBox, HTML
import ipywidgets as widgets
from IPython.display import display

def visualize(dataset, file, image_format="jpg", start_index=0, length=1000, jupyterNB=True, save_images=False):
    """
    Visualize the results with a styled slider below a centered image for Jupyter Notebooks.

    Args:
        dataset (str): Name of the dataset directory
        file (str): The path to the txt-file containing data of the images and icebergs.
        image_format (str): File formats of the images (file extension without the dot)
        start_index: The index of the first image to process. Default is 0 (process from the beginning).
        length (int): Number of images to visualize. Default is 1000.
        jupyterNB (bool): True if visualizing takes place in jupyter notebooks. Default is True.
        save_images (bool): Whether or not to save the images with bounding boxes drawn.
    """
    if file is None:
        image_dir = os.path.join(DATA_DIR, dataset, "images", "processed")
    else:
        image_dir = os.path.join(DATA_DIR, dataset, "images", "raw")
    image_format = f".{image_format}".lower()
    colormap = {}  # Store colors for each object ID

    # Load the tracking data
    if file is not None:
        det_data = pd.read_csv(file, header=None)
        det_data.columns = ['frame', 'ID', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'x', 'y', 'z']
    else:
        det_data = pd.DataFrame()

    # Get sorted list of image paths
    sorted_images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_format)
    ])

    # Trim to the desired range
    sorted_images = sorted_images[start_index:start_index + length]

    if jupyterNB == True:
        # Define a more styled slider
        styled_slider = IntSlider(
            min=0,
            max=len(sorted_images) - 1,
            step=1,
            value=0,
            description='Image Index:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%') # Make it wider
        )

        # Add some custom CSS for a more modern look
        display(HTML("""
            <style>
                .widget-label { font-weight: bold !important; }
                .widget-slider { -webkit-appearance: none; height: 15px; border-radius: 5px; background: #d3d3d3; outline: none; -webkit-transition: .2s; transition: opacity .2s; }
                .widget-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #4CAF50; cursor: pointer; }
                .widget-slider::-moz-range-thumb { width: 20px; height: 20px; border-radius: 50%; background: #4CAF50; cursor: pointer; }
                .output_subarea { text-align: center !important; } /* Center the image output */
            </style>
        """))

        def show_image(index):
            img_name = sorted_images[index]
            image_path = os.path.join(image_dir, img_name)
            img = cv2.imread(image_path)
            image_no = img_name[:-4]

            if file is not None:
                frame_data = det_data[det_data['frame'] == image_no]
            else:
                frame_data = pd.DataFrame()

            # Draw bounding boxes
            for _, row in frame_data.iterrows():
                object_id = int(row['ID'])
                if object_id not in colormap:
                    np.random.seed(object_id)
                    color = tuple(map(int, np.random.randint(0, 255, size=3)))
                    colormap[object_id] = color
                color = colormap[object_id]
                bbox_left, bbox_top = int(row['bbox_left']), int(row['bbox_top'])
                bbox_width, bbox_height = int(row['bbox_width']), int(row['bbox_height'])
                cv2.rectangle(img, (bbox_left, bbox_top), (bbox_left + bbox_width, bbox_top + bbox_height), color, 2)
                cv2.putText(img, str(object_id), (bbox_left, bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Show image
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(img_name)
            ax.axis('off')
            plt.close(fig)  # Prevent duplicate display

            return fig

        def display_image(index):
            fig = show_image(index)
            display(HTML(f"<div style='text-align: center;'>{fig_to_html(fig)}</div>"))

        def fig_to_html(fig):
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'
            return img_html

        import base64
        ui = VBox([HTML("<div style='text-align: center;'></div>"), styled_slider])
        out = widgets.interactive_output(display_image, {'index': styled_slider})
        display(ui, out)

    else:
        for i, img_name in enumerate(sorted_images):
            if i < start_index:
                continue  # Skip images before the start index
            if i - start_index > length:
                break

            if img_name.lower().endswith(image_format):
                image_path = f"{image_dir}/{img_name}"
                img = cv2.imread(image_path)
                image_no = img_name[:-4]  # Get the frame ID from image name

                if img is None:
                    print(f"Image {image_path} not found.")
                    return

                # Filter detections for the current frame
                if file is not None:
                    frame_data = det_data[det_data['frame'] == image_no]
                else:
                    frame_data = pd.DataFrame()

                # Draw bounding boxes and labels on the image
                for _, row in frame_data.iterrows():
                    object_id = int(row['ID'])

                    # Assign a random color to each object ID (if not already assigned)
                    if object_id not in colormap:
                        np.random.seed(object_id)  # Ensure consistent color assignment
                        color = tuple(map(int, np.random.randint(0, 255, size=3)))
                        colormap[object_id] = color
                    color = colormap[object_id]

                    # Extract bounding box coordinates
                    bbox_left, bbox_top = int(row['bbox_left']), int(row['bbox_top'])
                    bbox_width, bbox_height = int(row['bbox_width']), int(row['bbox_height'])

                    # Draw the bounding box and label on the image
                    cv2.rectangle(img, (bbox_left, bbox_top), (bbox_left + bbox_width, bbox_top + bbox_height), color, 2)
                    cv2.putText(img, str(object_id), (bbox_left, bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Display the image
                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(img_name)
                plt.axis('off')
                plt.show()

    # Save the image if required
    if save_images:
        output_path = os.path.join(DATA_DIR, dataset, "results", "images")
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")