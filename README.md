# MRI Brain Tumour Segmentation & Classification

## Project Overview

This project explores a two-stage deep learning pipeline for MRI brain
tumour analysis using **U-Net segmentation** and **CNN classification**.

The objective was to first identify relevant regions in MRI images using
U-Net and then classify the images into four tumour categories:

* Glioma
* Meningioma
* Pituitary
* No Tumour

The project was developed using Python and TensorFlow/Keras in Google Colab
with GPU acceleration.

## Objectives

* Preprocess and explore MRI brain images.
* Generate segmentation masks using a U-Net architecture.
* Use CNNs for multi-class MRI classification.
* Evaluate segmentation and classification performance.
* Investigate how segmentation outputs can be incorporated into classification.

## Dataset

The dataset contains four classes:

```text
archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Dataset distribution:

* Training: 5,712 images
* Testing: 1,311 images
* Classes: 4

The training data was further divided into training and validation sets.

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* OpenCV/PIL
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* GPU acceleration

## Methodology

### 1. Image Preprocessing

MRI images were converted to RGB, resized to `128 × 128` for segmentation,
and pixel values were normalised to the range `[0, 1]`.

For CNN classification, images were resized to `224 × 224`.

### 2. Initial Segmentation

A simple threshold-based method was used to create segmentation masks.

The dummy segmentation approach used the green colour channel and a threshold
of `0.5` to generate binary masks.

These masks were then used as the training targets for the U-Net model.

### 3. U-Net Segmentation

A simplified U-Net architecture was implemented with:

* Convolutional layers
* Max pooling
* Bottleneck layers
* Upsampling
* Skip connections
* Sigmoid output layer

The model was trained for 10 epochs using binary cross-entropy loss.

### 4. Segmentation Limitation

A major limitation of this project was the use of **dummy segmentation masks**.

Because the masks were generated using a basic colour-channel threshold
rather than manually annotated tumour regions, they did not accurately
represent the actual anatomical tumour boundaries.

Consequently, the U-Net learned to reproduce these artificial masks rather
than learn clinically meaningful tumour segmentation.

The U-Net reported high validation pixel accuracy, but this metric should not
be interpreted as evidence of accurate tumour segmentation.

This was an important finding in the project: **the quality of segmentation
targets directly affects the usefulness of the segmentation model.**

### 5. CNN Classification

A CNN was developed to classify MRI images into four categories:

```text
Glioma
Meningioma
No Tumour
Pituitary
```

The CNN architecture included:

```text
Conv2D (32)
MaxPooling

Conv2D (64)
MaxPooling

Conv2D (128)
MaxPooling

GlobalAveragePooling
Dense (128)
Dropout (0.5)
Softmax output
```

The model used the Adam optimiser and categorical cross-entropy loss.

## CNN Results

The CNN achieved:

| Metric              | Result |
| ------------------- | -----: |
| Validation Loss     | 0.8532 |
| Validation Accuracy | 69.94% |

The classification results were further evaluated using:

* Confusion matrix
* Precision
* Recall
* F1-score
* Training/validation curves

## Segmentation-to-Classification Pipeline

The project also experimented with using U-Net outputs as inputs to the CNN.

The workflow was:

```text
MRI Image
    ↓
U-Net
    ↓
Predicted Segmentation Mask
    ↓
Resize / Convert to RGB
    ↓
CNN
    ↓
Predicted Tumour Class
```

However, because the U-Net was trained using artificially generated masks,
the segmentation outputs did not reliably represent tumour regions.

Therefore, using these outputs as CNN inputs introduced an important
limitation: **errors and inaccuracies from the segmentation stage could be
propagated into the classification stage.**

The notebook demonstrates this dependency through the `predict_mri()`
pipeline.

## Key Finding

The project highlights an important deep learning principle:

> **A downstream model can be affected by errors introduced by an upstream
> model.**

In this case, inaccurate segmentation targets produced unreliable
segmentation outputs. When these outputs were subsequently used as inputs
for classification, the quality of the classification pipeline could also
be affected.

The results therefore demonstrate why reliable, domain-specific annotations
are important when developing medical image segmentation systems.

## Evaluation

The project includes visualisations for:

* Original MRI images
* Generated segmentation masks
* U-Net training performance
* CNN training/validation curves
* Confusion matrix
* Classification report
* Segmentation/classification comparison

## Project Structure

```text
MRI-Segmentation-Classification/
│
├── MRI images segmentation and classification.ipynb
├── README.md
└── dataset/
    ├── Training/
    └── Testing/
```

The MRI dataset is not included in the repository due to its size.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd MRI-Segmentation-Classification
```

### 2. Install dependencies

```bash
pip install tensorflow numpy pillow
pip install matplotlib seaborn scikit-learn
```

### 3. Open the notebook

Run:

```text
MRI images segmentation and classification.ipynb
```

The notebook was developed in Google Colab and can be run with GPU
acceleration.

## Key Learning Outcomes

This project demonstrates practical experience with:

* Medical image preprocessing
* Image segmentation
* U-Net architecture
* CNN classification
* TensorFlow/Keras
* Multi-class classification
* Model evaluation
* Confusion matrices
* Error propagation in ML pipelines
* GPU-based deep learning

## Limitations and Future Improvements

The main limitation was the absence of reliable, manually annotated tumour
segmentation masks.

Future improvements could include:

* Using expert-annotated segmentation masks.
* Training U-Net with clinically meaningful tumour boundaries.
* Using Dice loss or combined Dice/BCE loss.
* Applying stronger image preprocessing.
* Using transfer learning for classification.
* Comparing CNN classification using original images versus reliable
  segmented regions.
* Evaluating segmentation with Dice and IoU using genuine ground-truth masks.

## Conclusion

This project demonstrates a complete experimental pipeline combining image
segmentation and classification for MRI brain tumour analysis.

Although the U-Net achieved high pixel-level accuracy against the generated
masks, the use of dummy masks limited the clinical meaning of the
segmentation results. The project also showed how unreliable segmentation
outputs can influence downstream classification when they are used as CNN
inputs.

The CNN achieved **69.94% validation accuracy**, providing a useful baseline
for further experimentation with improved segmentation targets and
classification techniques.

## Author

**Maame Birago Aninkorah**

MSc Data Analytics
