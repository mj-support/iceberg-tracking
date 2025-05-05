# Iceberg Tracking

## Getting Started

### Installation

```bash
git clone https://github.com/mj-support/iceberg-tracking.git
cd iceberg-tracking
conda env create -f environment.yml
conda activate iceberg-tracking
```

## Usage

### Preparation

To get started, make sure your raw images are stored in the following directory structure: ```data/<dataset_name>/raw/```

To preprocess the data (e.g., brighten, mask, or tile the images) execute the provided ```preprocessing.py``` script. 

Additionally, you need a ```det.txt``` file containing iceberg detections for your dataset. This file can be generated manually (e.g., via labeling) or produced by a detection model.
Each line in this file should follow this format: 

```<image_without_extension>,<iceberg_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<unused_1>,<unused_2>,<unused_3>```

Example: ```_MG_17310_B,1,161.15,1087.59,247.32,90.46,0.9999,-1,-1,-1```


### Training

Execute each training run

1. get_coco_annotations.py: transform annotations to COCO format
2. model_train.py: Faster R-CNN model with a ResNet50-FPN backbone to detect icebergs
3. model_inference.py: Inference to get all detections
4. tracking.py: calculate distance and overlapping, match detections, filter outliers and nested icebergs, visualize results


## Glacier
- name: Equalorutsit Kangiliit Sermiat OR Qajuuttap Sermia 
- 3km wide calving front
- ice ~50-80m high at calving front
- coordinates: [61.332, -45.780](https://www.google.de/maps/place/61%C2%B019'55.2%22N+45%C2%B046'48.0%22W/@61.3045198,-45.8038482,7726m/data=!3m1!1e3!4m4!3m3!8m2!3d61.332!4d-45.78!5m1!1e4?entry=ttu&g_ep=EgoyMDI1MDIyNi4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D)

## Further steps
- add similarity score to match detections
- calculate area covered with ice
  - consider the shape of the iceberg and perspective, possibly work with pixel color values
- georeferencing: integrate both perspectives
- melange detection
- satellite perspective