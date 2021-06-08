# Brain Tumor Classifier
This repository contains project on Brain Tumor Classification that I made as a part of an online course on Coursera.

### Task
Given MRI scans of the brain, the goal is to identify the type of **brain tumor**.

The images belong to the following 4 classes:
1. Glioma Tumor
2. Meningioma Tumor
3. No Tumor
4. Pituitary Tumor

In essence, this is an **image classification problem**.

#### Samples from Tumor Classes
![alt text](https://github.com/kvarun07/brain-tumor-classifier/blob/main/assets/tumor_classes_sample.png)

### Approach
[EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) is an efficent model that reaches State-of-the-Art accuracy on image classification transfer learning tasks. In this problem, EfficientNet with weights pre-trained on *imagenet* are used.

### Model training curves
The performance metrics of the model with respect to number of epochs can be visualised as follows:
![alt text](https://github.com/kvarun07/brain-tumor-classifier/blob/main/assets/model_curves.png)

### Result
The model yielded > 95% accuracy when tested a large number of times.
