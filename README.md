# Severstal Steel Defect Detection

Jupyter notebook for the Severstal steel defect detection workflow. It trains a classifier, a full-image FPN segmenter, and a crop-based FPN refiner, then evaluates predictions and writes metrics/visual summaries.

The project implements a multi-stage deep learning pipeline for industrial steel surface defect localization using PyTorch and transfer learning.

## Project Highlights

- Binary image classifier for defect presence detection
- Full-image FPN semantic segmentation model
- ROI-based crop refinement network
- Transfer learning with a pretrained ResNet34 encoder
- Test-Time Augmentation (TTA)
- Threshold optimization
- Connected component filtering
- Morphological post-processing

## Results

| Split | Dice | IoU |
| --- | ---: | ---: |
| Validation | 80.17% | 73.04% |
| Test | 79.92% | 72.82% |

## Fast Review

If you only want to review the finished results, open:

```text
outputs_improved_strongest/final_project_summary.txt
```

You can also run the quick summary cell near the top of the notebook. It prints the saved summary without training or evaluating the models.

The full notebook run needs the dataset and trained checkpoint files. If the checkpoint files are missing, the training cells will start from scratch.

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
    ├── *_metrics.json
    ├── *_history.csv
    ├── *_summary.json
    ├── *_per_image_results.csv
    ├── threshold_search.csv
    └── visuals/
        └── *_prediction.png
```

## Setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the Severstal dataset and place it at:

   ```text
   ~/Downloads/severstal-steel-defect-detection
   ```

   Or set a custom dataset path:

   ```bash
   export SEVERSTAL_BASE_PATH="/path/to/severstal-steel-defect-detection"
   ```

3. Open `severstal_real_notebook_cells_mac_vscode.ipynb` in VS Code or Jupyter and run the cells from top to bottom.

## Full Run Behavior

The notebook preserves the trained pipeline behavior:

- If `best_cls.pth`, `best_seg.pth`, and `best_ref.pth` are available under `outputs_improved_strongest/checkpoints/`, the notebook loads them and skips training.
- If checkpoints are missing and `FORCE_RETRAIN = False`, the notebook trains the missing stages.
- `RESUME = True` allows interrupted training to continue from the latest checkpoint.
- The saved threshold in `best_threshold.json` is reused when available.

## Outputs

The notebook writes output files to `outputs_improved_strongest/`, including metrics, CSV summaries, threshold search results, visualizations, and model checkpoints.

Model checkpoint files (`.pth`, `.pt`, `.ckpt`) are ignored by Git because they are large. Store trained weights with Git LFS, a GitHub Release, Kaggle, Google Drive, or another artifact store if you want to share them.

## Notes

- The default encoder is `resnet34`, which works well for limited VRAM.
- For stronger experiments, try `timm-efficientnet-b3` and reduce `BATCH` if memory is tight.
- Dataset files are not included in this repository.
- Checkpoint files are not included in Git. Use Git LFS, a GitHub Release, Kaggle, Google Drive, or another artifact store to share trained weights.
