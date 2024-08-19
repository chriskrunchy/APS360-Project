# APS360 Project - University of Toronto
## Group 60 - Final Report: Efficient MultiLevelUNet for Chest X-ray Classification

This repository houses our project for APS360: Applied Fundamentals of Deep Learning, which entails the development and evaluation of an advanced deep learning model for classifying chest X-ray images into various disease categories.

### Project Overview
The field of medical imaging has greatly benefitted from the application of deep learning technologies. Our project focuses on leveraging these advancements to assist radiologists in diagnosing diseases more accurately and efficiently through an enhanced deep learning model capable of classifying chest X-ray images.

### Motivation
The primary motivation for this project is to improve diagnostic accuracy for lung diseases using machine learning, which could be particularly beneficial in areas with a shortage of medical professionals.

### Model Architecture: Efficient MultiLevelUNet
The Efficient MultiLevelUNet, adapted from the traditional U-Net architecture, incorporates several modifications and optimizations to tailor it for chest X-ray classification:

- **Downsampling and Bottleneck**: Captures deep contextual information and abstract features.
- **EfficientNet-Inspired Scaling**: Systematic scaling of network width, depth, and resolution enhances capacity and efficiency.
- **Global Average Pooling and Fully Connected Layers**: Aggregate spatial features and translate them into class scores.

#### Key Features
- **Architecture based on U-Net and EfficientNet** innovations to optimize feature extraction and processing capabilities.
- **Customized data handling** techniques to manage class imbalance and enhance model training efficiency.
- **Utilization of advanced scaling techniques** to balance computational efficiency with robust pattern recognition capabilities.

### Dataset
We employed the NIH Chest X-ray dataset, meticulously processing it to ensure high-quality data for training our models. The dataset comprises 85,113 X-ray images, balanced across various classes of lung diseases.

#### Data Processing Highlights
- **Initial Filtering**: Focus on images with a single disease label.
- **Class Balancing**: Address significant class imbalances through undersampling and oversampling.
- **Augmentation Techniques**: Enhance generalization through methods like cropping, rotation, and color jittering.

### Results and Discussion
Our Efficient MultiLevelUNet demonstrated superior performance over the baseline ResNet18 model, achieving higher accuracy and efficiency in classifying chest X-rays. The model effectively learned and generalized from the training data, showcasing particularly strong performance in critical disease classifications.

### Ethical Considerations
We emphasize the use of our model as an assistive tool in medical diagnostics to support, not replace, professional radiological assessments.

## Repository Contents
- `ChestX-Ray/`: Contains all files and scripts related to chest X-ray classification.
- `Data_processing_for_final_draft/`: Scripts for data preprocessing, augmentation, and setup for final model training.
- `OCT/`: Code and resources for OCT (Optical Coherence Tomography) image analysis. This dataset was later abandoned. 
- `basic-architectures/`: Basic architecture files for initial model setups and experiments.
- `data/`: Directory for dataset storage and organization. Please follow data setup instructions to properly organize this directory.
- `endoscopy/`: Resources and models specific to endoscopy image analysis.
- `resnet/`: Customized ResNet architectures used in the project.
- `unet-clf/`: U-Net models adapted for classification tasks.
- `unet-seg/`: U-Net models designed for segmentation tasks.
- `APS360_Project_Final_Report.pdf`: Detailed project report outlining the methodology, results, and conclusions.
- `APS360_Final_Video.mp4`: A video presentation summarizing the project and showcasing key findings.
- `APS_Final_data_processing_classification...`: Additional scripts for data processing and classification.
- `Final Presentation APS.pdf`: Presentation slides for the project summary and results.
- `README.md`: Provides an overview of the project, setup instructions, and additional documentation.


### Contributions
This project was a collaborative effort by Adam Roberge, Bill Jiang, Chris Kwon, and Mitchell Souliere-Lamb, under the guidance of our course instructors at the University of Toronto.

### Citation and References
If you find this work useful, please consider citing our project. More detailed references and related works can be found in our [final project report](APS360_Project_Final_Report.pdf) and [final video](APS360_Final_Video_2.mp4)

---
For more information, issues, or questions, please contact us through the repository issues or pull requests.

