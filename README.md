# Iceberg Tracking

This repository provides the implementation of detecting and tracking icebergs in glacier imagery captured during field observations.


<div align="center">
    <img src="data/hill_2min_2023-08/videos/tracking.gif" alt="hill" width="375"/>
    <img src="data/hill_2min_2023-08/videos/tracking_night.gif" alt="night" width="375"/>
</div>


### Glacier
- name: Equalorutsit Kangiliit Sermiat OR Qajuuttap Sermia 
- 3km wide calving front
- ice ~50-80m high at calving front
- coordinates: [61.332, -45.780](https://www.google.de/maps/place/61%C2%B019'55.2%22N+45%C2%B046'48.0%22W/@61.3045198,-45.8038482,7726m/data=!3m1!1e3!4m4!3m3!8m2!3d61.332!4d-45.78!5m1!1e4?entry=ttu&g_ep=EgoyMDI1MDIyNi4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D)

## Getting Started

### Installation

```bash
git clone https://github.com/mj-support/iceberg-tracking.git
cd iceberg-tracking
conda env create -f environment.yml
conda activate iceberg-tracking
```

### Preparation

#### Images
- To get started, make sure your raw images are stored in: ```data/<dataset_name>/raw/```

#### Training data
  - Input data with iceberg detections needs to be stored in: `data/{dataset_name}/annotations/gt.txt`
  - This file can be generated manually (e.g., via labeling) or produced by a detection model. 
  - It follows the MOT format where each line consists of the following 10 values based on the preprocessed images:
    - ```<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>```

## Basic Usage

```python
from preprocessing import ImagePreprocessor 
from detection import IcebergDetector
from tracking import IcebergTracker
from utils.visualize import Visualizer

dataset = "hill_2min_2023-08"
image_format = "JPG"

# Run the preprocessing pipeline with optional brightening, masking and tiling
preprocessor = ImagePreprocessor(dataset=dataset, image_format=image_format)
preprocessor.process_images()

# Train a Faster R-CNN model to detect icebergs + run inference on glacier images
detector = IcebergDetector(dataset=dataset, image_format=image_format)
detector.train()
detector.predict()

# Match and track the detected icebergs based on size, distance and intersection
tracker = IcebergTracker(dataset=dataset, image_format=image_format)
tracker.track()

# Visualize the tracking results
visualizer = Visualizer(dataset=dataset, image_format=image_format, stage="tracking")
visualizer.render_video()
```

## Further steps
- calculate area covered with ice
  - consider the shape of the iceberg and perspective, possibly work with pixel color values
- georeferencing: integrate both perspectives
- melange detection
- satellite perspective