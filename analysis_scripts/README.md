# Analysis Scripts

This directory contains the analysis scripts used to generate results for our Voice Activity Projection experiment.

## Pipeline Overview

The analysis pipeline consists of the following steps, executed in order:

### 1. Audio Preprocessing
**Script:** `combine_mono_to_stereo.py`

Combines initial mono recordings into stereo audio files compatible with the VAP model.

### 2. VAP Prediction
**Script:** `process_dataset.py` (located in project root)

Runs the VAP model to generate frame-wise probability predictions for conversations. Can process:
- Individual speaker pairs
- Entire dataset using the `--recursive` flag

Outputs VAP prediction results in JSON format for each conversation.

## Analysis Methods

### Prediction-Ground Truth Discrepancy

#### 3. Prediction Conversion
**Script:** `json_to_csv.py`

Converts JSON VAP outputs (frame-wise probabilities) into CSV format with discrete turn-taking predictions.
- Uses `p_now` probability threshold
- Applies hysteresis thresholding to determine turn timings

#### 4. Accuracy Evaluation
**Script:** `pred_gt_diff.py`

Compares predicted turns against ground truth annotations to evaluate model performance.

> **Note:** Ground truth data cannot be published due to data protection restrictions.

### Perspective Divergence

#### 5. Divergence Computation
**Script:** `compute_pairwise_divergency.py`

Computes Jensen-Shannon divergence between speaker perspectives from JSON VAP outputs. Results are saved as frame-wise divergence in JSON format for further analysis.

## Usage

Execute scripts sequentially following the numbered order above to reproduce the complete analysis pipeline. The data used to obtain these results is not open source however, so only the code can be seen here.