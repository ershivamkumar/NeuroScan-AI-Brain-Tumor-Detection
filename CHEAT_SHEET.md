# NeuroScan AI — Quick Reference Cheat Sheet

## 🚀 Quick Start Commands

```bash
# 1. Clone repository
git clone https://github.com/ershivamkumar/NeuroScan-AI-Brain-Tumor-Detection.git
cd NeuroScan-AI-Brain-Tumor-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch interactive web dashboard
streamlit run app.py

# 4. Command-line prediction (single image)
python inference.py --image pred/pred1.jpg

# 5. Batch command-line evaluation
python inference.py --folder pred/ --output batch_results.csv
```

---

## 📁 Repository Directory Structure

```text
NeuroScan-AI-Brain-Tumor-Detection/
├── app.py                             # Streamlit interactive application
├── inference.py                       # CLI inference & batch prediction tool
├── requirements.txt                   # Production & research dependencies
├── README.md                          # Comprehensive project documentation
├── USAGE_GUIDE.md                     # Step-by-step user & deployment manual
├── CHEAT_SHEET.md                     # Quick reference sheet
├── LICENSE                            # MIT License & medical disclaimer
├── Brain_Tumor_Detection_CNN_LSTM.ipynb # Full training, comparison & evaluation notebook
├── outputs/                           # Trained weights and benchmark metrics
│   ├── mobilenetv2_best.keras         # High-performing MobileNetV2 (94.44% acc)
│   ├── mobilenetv2_final.keras        # Final MobileNetV2 checkpoint
│   ├── best_model.keras               # CNN-LSTM baseline model (50.89% acc)
│   ├── final_model.keras              # Final CNN-LSTM checkpoint
│   ├── mobilenetv2_test_metrics.csv   # Real test evaluation metrics (MobileNetV2)
│   ├── test_metrics.csv               # Real test evaluation metrics (CNN-LSTM)
│   └── model_comparison.csv           # Direct model benchmark comparison table
├── yes/                               # 1,500 MRI scans with Brain Tumor (Class 0)
├── no/                                # 1,500 MRI scans Healthy / No Tumor (Class 1)
├── pred/                              # 60 Unlabeled test MRI scans
└── Br35H-Mask-RCNN/                   # Bounding box & mask annotations (TRAIN, TEST, VAL)
```

---

## 🎯 Model Architecture Overview

### 1. MobileNetV2 (Primary Production Model - 94.44% Accuracy)
```text
Input: 150×150×3 RGB MRI Image (preprocessed [-1, 1])
  ↓
[MobileNetV2 Backbone] Pretrained on ImageNet (weights frozen)
  ├─ 53 Inverted Residual Blocks (Depthwise Separable Convolutions)
  └─ Last Feature Map: `Conv_1_relu` / `out_relu` (used for Grad-CAM)
  ↓
[GlobalAveragePooling2D] (1280 feature maps → 1280-dim vector)
  ↓
[Dropout(0.30)]
  ↓
[Dense] 64 units, ReLU activation
  ↓
[Dropout(0.20)]
  ↓
[Dense] 2 units, Softmax activation → [Tumor (0), Healthy (1)]
```

### 2. CNN-LSTM (Baseline Experimental Model - 50.89% Accuracy)
```text
Input: 150×150×1 Grayscale MRI Image (normalized [0, 1])
  ↓
[CNN Block 1] Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.20)
[CNN Block 2] Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
[CNN Block 3] Conv2D(128, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.30)
  ↓
[Reshape] Sequence dimension (-1, 128)
  ↓
[LSTM] 64 recurrent units, recurrent dropout=0.20
  ↓
[Dense] 64 units, ReLU activation → Dropout(0.30)
  ↓
[Dense] 2 units, Softmax activation
```

---

## 📊 Benchmark Performance Comparison

| Metric | CNN-LSTM Baseline | MobileNetV2 Transfer Learning | Difference |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 50.89% | **94.44%** | **+43.55%** |
| **ROC-AUC** | 72.49% | **98.57%** | **+26.08%** |
| **Precision** | 66.89% | **95.45%** | **+28.56%** |
| **Recall** | 50.89% | **93.33%** | **+42.44%** |
| **F1-Score** | 35.65% | **94.38%** | **+58.73%** |
| **Correct Test Predictions** | 229 / 450 | **425 / 450** | **+196 images** |

---

## 🔧 Essential Code Snippets

### Preprocessing Single MRI
```python
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_mri(image_path: str, img_size: int = 150) -> np.ndarray:
    raw = Image.open(image_path)
    gray = ImageOps.grayscale(raw).resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    rgb = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.expand_dims(preprocess_input(rgb), axis=0)
```

### Loading Model & Predicting
```python
import tensorflow as tf

model = tf.keras.models.load_model("outputs/mobilenetv2_best.keras")
x = preprocess_mri("pred/pred1.jpg")
probs = model(x, training=False).numpy()[0]

classes = {0: "Brain Tumor", 1: "Healthy / No Tumor"}
predicted_class = int(np.argmax(probs))
print(f"Prediction: {classes[predicted_class]} ({probs[predicted_class]*100:.2f}%)")
```
