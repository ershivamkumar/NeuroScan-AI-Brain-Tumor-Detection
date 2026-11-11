# NeuroScan AI — User Guide & Execution Manual

## 1. Project Overview

**NeuroScan AI** is an advanced deep learning framework and interactive diagnostic research prototype for detecting brain tumors from Magnetic Resonance Imaging (MRI) scans. The project benchmarks two primary neural network architectures:
- **Transfer Learning with MobileNetV2**: High-accuracy deep convolutional feature extraction pretrained on ImageNet and fine-tuned for MRI scans (Achieved **94.44% Test Accuracy** and **98.57% ROC-AUC**).
- **Hybrid CNN-LSTM Architecture**: Spatial feature extraction via multi-layer 2D convolutions cascaded into Long Short-Term Memory (LSTM) recurrent sequence modeling (Achieved **50.89% Test Accuracy** and **72.49% ROC-AUC**).

The system integrates:
- **Interactive Streamlit Web Dashboard**: Single-scan analysis and side-by-side comparison mode.
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Visual explainability highlighting anatomical regions influencing inference.
- **Automated PDF Diagnostic Report Generation**: Structured clinical research summaries.
- **CLI Inference Pipeline**: Fast, programmatic single-image and batch predictions via `inference.py`.

---

## 2. Prerequisites

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version**: Python 3.10 or 3.11 recommended
- **Hardware**: CPU supported; CUDA-capable GPU recommended for high-throughput batch evaluation

---

## 3. Installation & Setup

### Step 1: Clone the Repository

Clone the project directly from GitHub:

```bash
git clone https://github.com/ershivamkumar/NeuroScan-AI-Brain-Tumor-Detection.git
```

### Step 2: Navigate to Project Directory

```bash
cd NeuroScan-AI-Brain-Tumor-Detection
```

### Step 3: Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Model Weights Configuration

Trained model weights are stored in the `outputs/` directory:
- `outputs/mobilenetv2_best.keras` (Primary high-performing model: 94.44% accuracy)
- `outputs/mobilenetv2_final.keras` (Full training checkpoint)
- `outputs/best_model.keras` (CNN-LSTM benchmark model: 50.89% accuracy)

The application automatically locates weights in `outputs/` or the root workspace directory.

---

## 5. Running the Application

### Option A: Launch Interactive Streamlit Web UI

Run the following command from the project root:

```bash
streamlit run app.py
```

Streamlit will start a local web server (typically at `http://localhost:8501`). Open this URL in your web browser.

#### Features in Streamlit Web UI:
1. **Analyze MRI Tab**:
   - Upload any JPG, JPEG, or PNG brain MRI scan.
   - Click **Analyze MRI** to trigger inference.
   - View predicted class (`Brain Tumor` or `Healthy / No Tumor`), confidence percentage, and comparative bar indicators.
   - Inspect Grad-CAM heatmaps and overlays.
   - Download a clinical research PDF report via **Download report (PDF)**.
2. **Compare MRIs Tab**:
   - Upload two separate MRI scans for independent, side-by-side comparative inspection.
   - View dual class predictions and Grad-CAM activations side-by-side.

### Option B: Command-Line Inference (CLI)

Run predictions directly from the terminal without launching the web server:

**Single Image Inference:**
```bash
python inference.py --image pred/pred1.jpg
```

**Single Image Inference with Specific Model:**
```bash
python inference.py --image yes/y100.jpg --model outputs/mobilenetv2_best.keras
```

**Batch Evaluation on a Folder:**
```bash
python inference.py --folder pred/ --output results.csv
```

### Option C: Jupyter Notebook Training & Experimentation

To inspect the data pipeline, train architectures, or reproduce benchmark metrics:

```bash
jupyter notebook Brain_Tumor_Detection_CNN_LSTM.ipynb
```

---

## 6. Class Mapping Reference

| Class Index | Label | Meaning |
|:---:|:---:|:---|
| **0** | `Brain Tumor` | Positive detection of abnormal brain lesion/tumor |
| **1** | `Healthy / No Tumor` | Scan shows no evidence of tumor tissue |

---

## 7. Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'reportlab'`
- Run: `pip install reportlab` or reinstall `requirements.txt`.

### Issue: TensorFlow GPU Detection
- If running on CUDA, ensure NVIDIA drivers (CUDA 12.x / cuDNN) are configured. For CPU-only execution, TensorFlow runs automatically without extra configuration.

### Issue: Streamlit Port Already in Use
- Run with a custom port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

## 8. Disclaimer

NeuroScan AI is an educational and research prototype. Model outputs are computational estimations and do not constitute a certified medical diagnosis or clinical treatment plan. Always consult certified healthcare specialists for clinical interpretations.
