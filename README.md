# 🍅 Tomato Leaf Disease Classification
 
> *Does preprocessing actually help — or just add noise?*
 
A CNN trained on [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) to classify four tomato leaf diseases, with a focus on comparing preprocessing strategies before training.
 
---
 
## The experiment
 
Four conditions, same architecture, same data:
 
| Condition | What it does |
|-----------|-------------|
| **Baseline** | Raw images, no preprocessing |
| **Segmentation** | HSV masking to isolate leaf tissue |
| **Augmentation** | Random transforms — rotation, zoom, flip, brightness |
| **Seg + Aug** | Both combined |
 
---
 
## Results (TL;DR)
 
**Augmentation wins.** Segmentation helps marginally. Combining both is the worst outcome — brightness augmentation breaks the HSV thresholds and the model gets garbage masks.
 
---
 
## Architecture
 
Custom CNN · 4 conv blocks (32 → 64 → 128 → 256) · BatchNorm + Dropout · SGD with momentum · EarlyStopping
 
```
Input (128×128×3)
  → Conv2D(32) → BN → MaxPool → Dropout(0.2)
  → Conv2D(64) → BN → MaxPool → Dropout(0.3)
  → Conv2D(128) → BN → MaxPool → Dropout(0.4)
  → Conv2D(256) → BN → MaxPool → Dropout(0.4)
  → Dense(256, L2) → BN → Dropout(0.5)
  → Dense(4, softmax)
```
 
---
 
## Classes
 
- Bacterial Spot  
- Yellow Leaf Curl Virus  
- Late Blight  
- Leaf Mold  
---
 
## Setup
 
```bash
pip install tensorflow kagglehub split-folders opencv-python
```
 
Open `tomato_disease_classification.ipynb` in Kaggle or Colab with GPU enabled.  
All hyperparameters live in the **Global Configuration** cell — one place, everything tuneable.
 
---
 
## Key finding
 
> On datasets where the subject already dominates the frame, augmentation alone is the most reliable strategy. Segmentation adds complexity without proportional gain — and actively hurts when combined with color-altering augmentations.
 
---
 
## Stack
 
`TensorFlow` · `OpenCV` · `scikit-learn` · `Matplotlib` · `Seaborn` · `KaggleHub`
