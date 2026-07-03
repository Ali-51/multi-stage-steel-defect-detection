# Multi-Stage Steel Defect Detection

## Overview

This repository contains a completed deep learning project for industrial steel surface defect detection. The project uses a multi-stage PyTorch pipeline to detect whether a steel image contains a defect and then localize the defective region with semantic segmentation.

The workflow is implemented in a Jupyter notebook and includes saved experiment outputs, metrics, threshold search results, and prediction visualizations. The goal is accurate defect localization, especially for small and difficult surface defects.

## Features

- Binary image-level defect classifier
- Full-image FPN semantic segmentation model
- ROI-based crop refinement network
- Transfer learning with a pretrained ResNet34 encoder
- Test-Time Augmentation (TTA)
- Validation-based threshold optimization
- Connected component filtering
- Morphological post-processing
- Saved metrics, summaries, CSV logs, and prediction visualizations
- Fast review cells for viewing saved results without training

## Pipeline

The project uses a three-stage inference pipeline:

| Stage | Component | Purpose |
| --- | --- | --- |
| Stage 0 | Binary classifier | Predicts whether an image contains a defect. |
| Stage 1 | Full-image FPN segmenter | Produces a coarse defect mask using global image context. |
| Stage 2 | Crop FPN refiner | Refines the predicted ROI to improve local defect boundaries. |

Final predictions combine the coarse full-image mask and the refined crop mask. Small false-positive components are filtered, and light morphological post-processing is applied before evaluation.

```text
Input image
    |
    v
Binary classifier
    |
    v
Full-image FPN segmentation
    |
    v
ROI extraction
    |
    v
Crop-based FPN refinement
    |
    v
Post-processing and final mask
```

## Results

The reported results are preserved from the completed experiment.

| Split | Dice | IoU |
| --- | ---: | ---: |
| Validation | 80.17% | 73.04% |
| Test | 79.92% | 72.82% |

Additional saved summary:

```text
Best threshold: 0.6000000000000001
Best threshold validation Dice: 0.7487
Test average inference time: 330.19 ms/image
```

## Technologies

| Category | Tools |
| --- | --- |
| Language | Python |
| Deep Learning | PyTorch, TorchVision |
| Segmentation | Segmentation Models PyTorch |
| Computer Vision | OpenCV |
| Augmentation | Albumentations |
| Data Processing | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Experiment Interface | Jupyter Notebook, Matplotlib, tqdm |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── severstal_real_notebook_cells_mac_vscode.ipynb
└── outputs_improved_strongest/
    ├── README.md
    ├── final_project_summary.txt
    ├── final_project_summary.json
    ├── best_threshold.json
    ├── best_threshold.txt
    ├── classifier_metrics.json
    ├── segmenter_metrics.json
    ├── refiner_metrics.json
    ├── classifier_history.csv
    ├── segmenter_history.csv
    ├── refiner_history.csv
    ├── threshold_search.csv
    ├── val_summary.json
    ├── test_summary.json
    ├── val_per_image_results.csv
    ├── test_per_image_results.csv
    └── visuals/
        └── *_prediction.png
```

## Quick Review

If you only want to review the completed results, open:

```text
outputs_improved_strongest/final_project_summary.txt
```

The notebook also includes fast review cells near the top:

1. `Quick saved-summary view`
2. `Quick saved-visuals view`

These cells display saved metrics and prediction images without training or re-running evaluation.

## Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Download the Severstal dataset and place it at:

```text
~/Downloads/severstal-steel-defect-detection
```

Or set a custom dataset path:

```bash
export SEVERSTAL_BASE_PATH="/path/to/severstal-steel-defect-detection"
```

Open the notebook:

```text
severstal_real_notebook_cells_mac_vscode.ipynb
```

Run the cells from top to bottom for the full pipeline.

## Full Run Behavior

The notebook preserves the trained pipeline behavior:

- If `best_cls.pth`, `best_seg.pth`, and `best_ref.pth` are available under `outputs_improved_strongest/checkpoints/`, the notebook loads them and skips training.
- If checkpoints are missing and `FORCE_RETRAIN = False`, the notebook trains the missing stages.
- `RESUME = True` allows interrupted training to continue from the latest checkpoint.
- The saved threshold in `best_threshold.json` is reused when available.

Checkpoint files are not tracked in Git because they are large. Store trained weights with Git LFS, a GitHub Release, Kaggle, Google Drive, or another artifact store if you want to share them.

## Outputs

The notebook writes generated outputs to:

```text
outputs_improved_strongest/
```

This folder includes:

- Final project summaries
- Per-stage metrics
- Training history CSV files
- Threshold search results
- Validation and test per-image results
- Saved prediction visualizations

## Future Improvements

- Add experiment configuration files for easier hyperparameter tracking.
- Provide downloadable model checkpoints through GitHub Releases or external artifact storage.
- Add a lightweight inference-only script for running predictions outside the notebook.
- Add automated tests for RLE decoding, mask generation, thresholding, and post-processing utilities.
- Compare additional encoders such as EfficientNet or ConvNeXt under the same validation protocol.
- Package the pipeline into reusable Python modules while preserving the notebook as the experiment report.

## Citation

If you use this project or build on it, please cite the repository:

```bibtex
@misc{multi_stage_steel_defect_detection,
  title = {Multi-Stage Steel Defect Detection},
  author = {Ali},
  year = {2026},
  url = {https://github.com/Ali-51/multi-stage-steel-defect-detection}
}
```

## Notes

- Dataset files are not included in this repository.
- Model checkpoint files are not included in Git.
- The default encoder is `resnet34`, which works well for limited VRAM.
- For stronger experiments, try `timm-efficientnet-b3` and reduce `BATCH` if memory is tight.
