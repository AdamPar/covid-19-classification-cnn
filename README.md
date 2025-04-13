# 🦠 COVID-19 X-ray Detection with CNN and Flask

This project uses a Convolutional Neural Network (CNN) to detect COVID-19 from grayscale chest X-ray images. It includes:

- A model training pipeline with TensorFlow/Keras
- Evaluation reports and visualizations
- A Flask web interface for live predictions

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AdamPar/covid-19-classification-cnn.git
cd covid-19-classification-cnn
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Setup

Download the dataset from Kaggle:  
[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Extract the dataset and place it in the root directory like this:

```
COVID-19_Radiography_Dataset/
├── COVID/
│   └── images/
├── Normal/
│   └── images/
```

---

## 🧠 Train the Model

Run the training script:

```bash
python train_model.py
```

This will:

- Train a CNN model
- Save it to `results/models/covid_cnn_model.h5`
- Generate:
  - Classification report (`results/raports/classification_report.txt`)
  - Confusion matrix
  - Accuracy/Loss plot
  - Precision/Recall plot

---

## 🌐 Run the Web App

Navigate to the Flask app directory:

```bash
cd flask-app
```

Run the app:

```bash
python app.py
```

Then open your browser to:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖼 Web Interface Features

- Upload and classify X-ray images
- See prediction results with confidence score
- Visualizations:
  - Confusion matrix
  - Accuracy & loss graph
  - Precision & recall graph
- Classification report display

---
