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

# Architecture 2: ResNet Inspired Image Classification Architecture

## Overview
This repository houses the implementation of a ResNet-inspired deep learning model specifically tailored for image classification tasks. The model leverages the deep residual learning framework to effectively handle the vanishing gradients problem, enabling the training of much deeper networks than was previously feasible.

## Model Architecture
Originally designed for image recognition, ResNet (Residual Network) uses shortcut connections or skip connections to skip one or more layers. Our adaptation, referred to as `ResNet_Classification`, modifies the conventional ResNet architecture to enhance its applicability for classifying images into multiple categories.

### Key Modifications
- **Residual Blocks**: Enhanced to extract features while maintaining the network's depth, essential for learning complex patterns.
- **Adaptive Pooling**: Placed before the final classification layer to ensure feature summarization is effective regardless of image dimensions.
- **Classification Layer**: Transforms the pooled features into final class scores, adapting the network from its original image recognition task to a broader classification capability.

## Dataset
The adapted ResNet model is trained and validated using the [RSNA 2024 Lumbar Spine Degenerative Classification Challenge dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) available on Kaggle. This dataset provides an excellent opportunity to apply deep learning for medical imaging classification.

### Dataset Details
- **Images**: MRI scans of the lumbar spine.
- **Labels**: Various degenerative conditions of the spine.
- **Objective**: Accurately classify each scan into predefined categories based on the degenerative state.
