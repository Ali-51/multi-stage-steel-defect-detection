# Saved Outputs

This folder contains the saved metrics, summaries, threshold search output, and visual examples from the completed experiment.

## Quick Review Files

- `final_project_summary.txt`: concise text summary of the final validation and test results.
- `final_project_summary.json`: structured version of the final summary.
- `visuals/`: saved prediction visualizations.

## Metrics and Logs

- `classifier_metrics.json`
- `segmenter_metrics.json`
- `refiner_metrics.json`
- `classifier_history.csv`
- `segmenter_history.csv`
- `refiner_history.csv`
- `threshold_search.csv`
- `val_summary.json`
- `test_summary.json`
- `val_per_image_results.csv`
- `test_per_image_results.csv`

## Checkpoints

Model checkpoint files are not tracked in Git because they are large. To reproduce the fast checkpoint-based run, place the trained checkpoint files under:

```text
outputs_improved_strongest/checkpoints/
```
