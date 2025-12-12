# Iceberg Tracking

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)

This repository provides the implementation of an iceberg detecting and tracking pipeline in time-lapse glacier imagery. 

---

## Overview:

The system combines: 
- **Faster R-CNN** for robust iceberg detection
- **Vision Transformer embeddings** for appearance-based similarity matching
- **Kalman filtering** for motion prediction and trajectory smoothing
- **Hybrid tracking algorithm** combining appearance, motion, and spatial features

### Key Features
- Detection of icebergs in challenging conditions (variable lighting, weather, dense fields)
- Multi-object tracking with ID consistency across extended time-lapse sequences
- Handles 2,000-3,000 icebergs per frame in dense calving events
- Achieves ~60% HOTA tracking performance
- Production-ready system with built-in visualization and segmentation features
- Flexible framework applicable to various glacier monitoring scenarios
- Get started quickly with pretrained iceberg detection and embedding models or train from scratch using your own data

---
<div align="center">
    <img src="examples/hill_2min_2023-08_tracking_0-9.gif" alt="hill" width="375"/>
    <img src="examples/hill_2min_2023-08_tracking_200-209.gif" alt="night" width="375"/>
</div>
<div align="center">
    <img src="examples/fjord_2min_2023-08_tracking_420-429.gif" alt="hill" width="375"/>
    <img src="examples/fjord_2min_2023-08_tracking_0-9.gif" alt="night" width="375"/>
</div>

**Glacier:** Equalorutsit Kangiliit Sermiat OR Qajuuttap Sermia 
- **Calving front:** ~3 km wide
- **Ice height:** 50-80 m at calving front
- **Location:** [61.332°N, 45.780°W](https://www.google.de/maps/place/61%C2%B019'55.2%22N+45%C2%B046'48.0%22W/@61.3045198,-45.8038482,7726m/data=!3m1!1e3!4m4!3m3!8m2!3d61.332!4d-45.78!5m1!1e4?entry=ttu&g_ep=EgoyMDI1MDIyNi4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D) (Greenland)
- **Imagery:** Time-lapse cameras capturing calving events and iceberg drift
- **Data source**: [GreenFjord Project](https://greenfjord-project.ch) - A research initiative studying glacier dynamics and calving processes in Greenland

---
## Installation

```bash
git clone https://github.com/mj-support/iceberg-tracking.git
cd iceberg-tracking
conda env create -f environment.yml
conda activate iceberg-tracking
```

---
## Quick Start

### Directory Structure

```
data/
└── <dataset_name>/
    ├── train/
    │   ├── images/           # Training images
    │   └── ground_truth/
    │       └── gt.txt        # MOT format annotations
    └── test/
        └── images/           # Test images (no annotations needed)
```

**Annotation Format**: Annotations in `gt.txt`, follow the [MOT Challenge](https://motchallenge.net/instructions/) format (CSV): Each line represents one object instance, containing ten comma-separated values: \
``<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>``

Example: \
``_MG_17310,1,161.15,1087.59,247.32,90.46,1.0,1,-1,-1``\
``_MG_17310,2,211.04,2085.49,248.25,97.31,1.0,1,-1,-1``

---

### Basic Usage

Run the pipeline using CLI commands:
```bash
# Train embedding model
python scripts/run_pipeline.py train-embedding dataset=hill/train

# Train detection model
python scripts/run_pipeline.py train-detection dataset=hill/train

# Run detection on test set
python scripts/run_pipeline.py detect dataset=hill/test

# Track icebergs across frames
python scripts/run_pipeline.py track dataset=hill/test

# Visualize tracking results
python scripts/run_pipeline.py visualize dataset=hill/test
```

---

For a step-by-step walkthrough with more detailed explanations and a Python-level interface example see [docs](docs/tracking-pipeline.ipynb).


**Happy tracking!**