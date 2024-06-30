# APS360-Project
Group project for APS360: Applied Fundamentals of Deep Learning

# Architecture 1: U-Net Inspired Image Classification Architecture

## Overview
This repository contains the implementation of a U-Net inspired deep learning model for image classification. The model architecture is adapted from the traditional U-Net model, which is primarily used for image segmentation tasks. Our adaptation modifies the U-Net architecture to make it suitable for classifying images into multiple categories.

## Model Architecture
U-Net is renowned for its effectiveness in biomedical image segmentation due to its ability to capture spatial hierarchies for pixel-level predictions. Our adaptation, named `UNET_Classification`, modifies the original architecture to cater to image classification tasks by integrating global average pooling and a fully connected layer at the end of the network. This approach allows the model to leverage the spatial feature extraction capabilities of U-Net while providing robust feature summaries for classification.

### Key Modifications
- **Downsampling Path**: Captures context and reduces spatial dimensions, increasing the receptive field.
- **Bottleneck**: This remains as the deepest part, aiming to extract the most abstract features.
- **Upsampling Path**: Instead of restoring original dimensions, it integrates skip connections and reduces to a feature-rich representation suitable for classification.
- **Global Average Pooling**: After upsampling, this layer reduces each feature map to a single value, effectively summarizing the spatial features.
- **Fully Connected Layer**: Transforms the pooled features into final class scores for classification purposes.

## Dataset
The model is trained and evaluated on the [RSNA 2024 Lumbar Spine Degenerative Classification Challenge dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) hosted on Kaggle. This competition focuses on the classification of spinal MRI scans into several degenerative categories, making it an excellent use case for our adapted U-Net model.

### Dataset Details
- **Images**: Spinal MRI scans.
- **Labels**: Multiple degenerative categories of the lumbar spine.
- **Objective**: To classify each MRI scan into the correct category based on visual degenerative features.

## Usage
To use this repository, clone it and install the required dependencies as listed in `requirements.txt`. Follow the instructions in `train.py` to start training the model on the RSNA dataset. For inference, use `predict.py` by providing the path to your MRI scan image.

## Contributions
Contributions to this project are welcome. You can contribute by improving the model's architecture, optimizing training procedures, or suggesting better data augmentation techniques.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
