# ImageNet 1k Image Classification with ResNet 50


## Model Architecture


## Data Augmentations

To enhance the model's robustness and generalization capabilities, we apply a series of data augmentations to the training dataset. These augmentations are inspired by the original ResNet paper and implemented using the albumentations library. The augmentations include random resized cropping, horizontal flipping, and color jittering, followed by normalization. These transformations help the model learn invariant features and improve performance on unseen data.

### Augmentations and Hyperparameters

1. **Random Resized Crop:**
   - Height: 224
   - Width: 224
   - Scale: (0.08, 1.0)
   - Aspect Ratio: (3/4, 4/3)
   - Probability: 1.0

2. **Horizontal Flip:**
   - Probability: 0.5

3. **Color Jitter:**
   - Brightness: 0.4
   - Contrast: 0.4
   - Saturation: 0.4
   - Hue: 0.1
   - Probability: 0.8

4. **Normalization:**
   - Mean: (0.485, 0.456, 0.406)
   - Standard Deviation: (0.229, 0.224, 0.225)

These augmentations are applied only to the training dataset, while the test dataset undergoes resizing and normalization to ensure consistent evaluation metrics.


## Model Results


