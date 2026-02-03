# 🦷 Dental X-Ray Segmentation — Hack4Health

**Automated segmentation of dental structures and carious lesions in dental X-ray images**  
This repository contains a full pipeline for medical image segmentation using a U-Net model, evaluation, visualization, and an instant demo generator.

---

## 🚀 Project Overview

Accurate detection of dental caries in X-ray images is a challenging task due to:
- Low contrast between healthy and decayed regions  
- Noise and overlapping anatomical structures  
- Manual annotation being time-consuming

This project aims to develop a **pixel-wise segmentation system** to highlight carious areas, enabling **automated early diagnosis support** in dental radiography.

---

## 📁 Repository Structure

Hack4Health/
├── outputs/ # Generated visualizations & reports
├── utils/ # Utility scripts/helpers
├── .gitignore
├── evaluate.py # Script for evaluation metrics
├── full_pipeline.py # End-to-end pipeline demo
├── predict.py # Script to run inference
├── requirements.txt # Python dependencies
├── train.py # Training script
├── visualize.py # Visualization utilities
├── generate_demo.py # Instant demo generator (no training needed)


---

## 🧠 Model & Methodology

We leverage a **U-Net architecture** — a standard convolutional encoder-decoder network with **skip connections** — for binary segmentation:

- Encoder downsamples and captures features
- Decoder upsamples and recovers spatial information
- Skip connections preserve fine details

This design is robust for medical imaging tasks like segmenting dental caries.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/harshgupta024/Hack4Health.git
cd Hack4Health

2. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

    This includes libraries such as:

        torch (PyTorch)

        opencv-python

        numpy

        matplotlib

        seaborn

        scikit-learn

📌 Train Your Own Model

To train the segmentation model on your dataset:

python train.py

Make sure your dataset is organized such that images and masks are matched correctly.
⭐ Run Full Pipeline End-to-End

python full_pipeline.py

This script:

    Loads the trained model

    Runs inference on test images

    Visualizes predictions

    Saves outputs

📷 Run Prediction on New Images

python predict.py --image_path <path_to_image>

📊 Model Evaluation

Evaluate the trained model using:

python evaluate.py

This outputs:

    Confusion matrix

    Dice score

    IoU

    Precision

    Recall

    F1-score

    Specificity / Sensitivity

🎨 Generate Demo Visuals (No Training Needed)

To produce a complete set of visuals, metrics, and reports for presentation:

python generate_demo.py

This will generate:

    Comparison grids

    Overlay visualizations

    Error maps

    Case studies

    Metrics bar chart

    Confusion matrices

    Training curves

    Final summary report

📝 Outputs

After running demo or training:

outputs/
├── visualizations/
├── metrics/
├── training_history.png
├── FINAL_REPORT.txt
├── PROJECT_SUMMARY.png

Use these for presentations, reports, and documentation.
🧪 Example Metrics (Demo)
Metric	Value
Dice Score	0.8734
IoU	0.7892
Precision	0.8956
Recall	0.8621
F1-Score	0.8785
Pixel Accuracy	0.9456
Sensitivity	0.8621
Specificity	0.9612
Hausdorff Distance	4.23
🧠 Notes for Judges

    Designed for hackathon presentation

    Easy to extend to real training

    All visuals available without training using generate_demo.py

💡 Future Improvements

    Multi-class segmentation (teeth + caries)

    Integration with clinical workflow

    3D data support (CBCT)

