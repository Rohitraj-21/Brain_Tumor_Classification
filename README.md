# 🧠 Brain Tumor Classification using ResNet50

This project focuses on classifying brain MRI images into four categories: **glioma**, **meningioma**, **notumor**, and **pituitary** using a fine-tuned ResNet50 deep learning model.

## 📁 Dataset

- Dataset used: [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- The dataset is organized into `Training/` and `Testing/` directories with class-wise subfolders.

## 📌 Key Features

- Uses **transfer learning** with a pretrained ResNet50 model.
- Custom classifier head with batch normalization and dropout for regularization.
- Applies **early stopping** to prevent overfitting.
- Visualizations include:
  - Sample images
  - Loss and accuracy curves
  - Confusion matrix

## 🛠️ Installation

Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Rohitraj-21/Brain_Tumor_Classification.git
   cd brain-tumor-classification
   ```

2. Download the dataset from Kaggle (automatically done using `kagglehub`).

3. Run the Python script:

   ```bash
   python brain_tumor_classification.py
   ```

4. (Optional) Run the notebook:
   Open `brain_tumor_classification.ipynb` in Google Colab or Jupyter.

## 📈 Results

- Achieves high accuracy on validation and test sets.
- Plots accuracy/loss over epochs.
- Displays confusion matrix for model evaluation.

## 📦 Output

- Trained model saved as: `best_model.pth`

## 🧠 Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary

## 📚 Dependencies

See `requirements.txt`.
