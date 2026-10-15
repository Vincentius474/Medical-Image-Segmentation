# CNN for Medical Image Segmentation

A convolutional neural network (U-Net-style encoder–decoder with skip connections) that segments lesions in grayscale medical images and labels each pixel as **normal**, **benign**, or **malignant**. Built with TensorFlow/Keras.

- **Author:** Reatlehile Vincent Sello
- **Module:** Machine Learning (Final Semester Project)
- **Notebook:** `CNN__Medical_Image_Segmentation_.ipynb`

---

<img width="940" height="754" alt="screenshotB" src="https://github.com/user-attachments/assets/c3a86a24-b1f6-436d-8aab-c9e2f8ffe7f3" />
<img width="940" height="1193" alt="screenshotA" src="https://github.com/user-attachments/assets/77fd574e-db5d-48a3-b039-f1e36efc2cc4" />

## Overview

The notebook covers the full pipeline:

1. Load images and their segmentation masks from class folders
2. Preprocess (resize, normalise, encode masks)
3. Build and train a segmentation CNN
4. Plot training/validation accuracy
5. Predict on a held-out test set and visualise results

## Dataset

The notebook expects a folder with one sub-directory per class:

```
Dataset/
├── normal/
│   ├── image_1.png
│   ├── image_1_mask.png
│   └── ...
├── benign/
│   └── ...
└── malignant/
    └── ...
```

- Each image is a `.png`, and its mask has the same name with a `_mask` suffix (e.g. `image_1.png` → `image_1_mask.png`).
- Images without a matching mask are skipped.
- The data is not included in this repository. Add your own dataset in the layout above.

### Preprocessing

| Step | Detail |
| --- | --- |
| Colour mode | Grayscale |
| Image size | Resized to 128 × 128 |
| Image scaling | Pixel values normalised to [0, 1] |
| Mask resizing | Nearest-neighbour interpolation (keeps labels intact) |
| Mask labels | `0` = normal/background, `1` = benign, `2` = malignant (any non-zero mask pixel gets its folder's class label) |
| Mask encoding | One-hot encoded, 3 classes |
| Train/test split | 80 / 20, `random_state=42` |

## Model Architecture

A U-Net-style network named `SegmentationCNN`, with input shape `(128, 128, 1)` and a 3-channel softmax output.

- **Encoder:** 3 blocks of Conv2D (3×3) → BatchNorm → ReLU → MaxPool, with 32 → 64 → 128 filters
- **Bridge:** Conv2D (128 filters) → BatchNorm → ReLU
- **Decoder:** 3 blocks of Conv2DTranspose (stride 2) → skip-connection concatenation → Conv2D → BatchNorm → ReLU, with 128 → 64 → 32 filters
- **Output:** 1×1 Conv2D with 3 filters and softmax activation

## Training Configuration

| Setting | Value |
| --- | --- |
| Optimizer | Adam (default settings) |
| Loss | Categorical cross-entropy |
| Metric | Accuracy |
| Epochs | 30 |
| Batch size | 10 |
| Validation | 20% of the training set (`validation_split=0.2`) |

## Requirements

- Python 3.8+
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- scikit-learn

Install with:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## Usage

The notebook was written for **Google Colab** and mounts Google Drive to read the dataset.

### On Google Colab

1. Upload your dataset to Google Drive.
2. Update `base_path` in the data-loading cell to point to it:
   ```python
   base_path = "/content/drive/My Drive/<your-path>/Dataset"
   ```
3. Run all cells.

### Locally

1. Remove or skip the `google.colab` Drive-mounting cell.
2. Set `base_path = "Dataset"` (a commented-out line is already in the notebook).
3. Run the notebook in Jupyter.

## Outputs

- **Model summary:** layer-by-layer printout of the architecture
- **Accuracy plot:** training vs. validation accuracy per epoch
- **Prediction visualisation:** for 5 test samples, shows the input image, ground-truth mask, and predicted mask side by side

## Project Structure

```
.
├── CNN__Medical_Image_Segmentation_.ipynb   # Full pipeline
├── README.md
└── Dataset/                                 # Not included; add your own
```

## Possible Improvements

- Add segmentation-specific metrics such as Dice coefficient and IoU. Pixel accuracy can look high when background dominates the image.
- Use a Dice or focal loss to handle class imbalance.
- Add data augmentation (flips, rotations, brightness shifts).
- Add early stopping and model checkpointing.
- Save the trained model for reuse.

## License
MIT
