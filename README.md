# 🐱🐶 Cat and Dog Image Classification using Support Vector Machine (SVM)

## 📌 Project Overview
This project demonstrates a machine learning approach to classify images of cats and dogs using a **Support Vector Machine (SVM)** model. The classification is based on Histogram of Oriented Gradients (HOG) features extracted from grayscale images. This project is part of the Prodigy InfoTech Machine Learning Internship.

### ✨ Key Objectives
- Implement image classification using SVM for the Kaggle Cats vs Dogs dataset.
- Apply various data preprocessing techniques and feature extraction methods.
- Train and evaluate the SVM model to achieve optimal accuracy.
- Experiment with hyperparameter tuning to enhance model performance.
- Demonstrate real-time predictions on new images.

## 📁 Dataset
The project utilizes the **Kaggle Cats vs Dogs Dataset**, which contains a total of **25,000 labeled images** of cats and dogs for training. The images are split into two categories:
- **Training Set**: 12,500 images of cats and 12,500 images of dogs.
- **Testing Set**: A subset of images from `test1.zip` is used for evaluating model performance.

Dataset Link: [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

## 🚀 Technologies Used
The following libraries and tools were utilized to build the project:

1. **Python**: Core programming language used for implementation.
2. **NumPy**: Numerical operations and data handling, especially for preprocessing image data.
3. **OpenCV (cv2)**: Image reading and preprocessing operations.
4. **scikit-image**: Used for HOG (Histogram of Oriented Gradients) feature extraction.
5. **scikit-learn**: Provided tools for data preprocessing, SVM model training, grid search, and evaluation metrics.
6. **Matplotlib**: Visualization of images and data.

## 🔍 Code Overview

### 1. **Dataset Loading**
Images are loaded from the dataset path and assigned corresponding labels for classification (0 for cat, 1 for dog).

### 2. **Image Preprocessing**
Each image is resized to a fixed dimension of **64x64 pixels** and converted to grayscale. HOG features are then extracted, reducing the dimensionality while preserving the structural information of images.

### 3. **Train-Test Split**
The preprocessed images are split into training and testing sets for model training and evaluation.

### 4. **SVM Model Training and Hyperparameter Tuning**
A Support Vector Machine (SVM) model is chosen for classification. A **grid search** is performed to determine the best hyperparameters (`C` and `kernel`) using cross-validation on the training set. The optimal parameters identified are:
- `C = 10`
- `kernel = 'rbf'`

### 5. **Model Evaluation**
The trained SVM model is evaluated on the test set, yielding an accuracy of approximately **83.17%**. A confusion matrix and classification report are generated to provide further insights into the model's performance.

### 6. **Image Prediction**
A separate function, `preprocess_image`, is defined to preprocess and predict the class of a single input image. The original and preprocessed images are visualized, and the predicted label is displayed using Matplotlib.

### 🔧 **Best Parameters**
- **Accuracy**: ~83.17%
- **Optimal Hyperparameters**: `C = 10`, `kernel = 'rbf'`

## 📈 Results & Insights
The SVM model achieved a satisfactory accuracy level, effectively differentiating between cat and dog images. Potential improvements can be made by experimenting with advanced feature extraction techniques or switching to deep learning models like Convolutional Neural Networks (CNNs).

## 🔮 Future Enhancements
- Experiment with different kernels (e.g., polynomial, sigmoid) and regularization parameters for better model performance.
- Implement Convolutional Neural Networks (CNNs) to automatically learn features directly from image data.
- Integrate additional feature extraction techniques to improve classification accuracy.

## 🖇️ How to Run the Project

1. **Download the Dataset**: Obtain the Kaggle Cats vs Dogs dataset from [this link](https://www.kaggle.com/c/dogs-vs-cats/data).
2. **Clone the Repository**:
   ```bash
   git clone [https://github.com/adnaanzshah/PRODIGY_ML_03]

Run the Jupyter Notebook: Open `Cats_Dogs_ml.ipynb` in Jupyter Notebook and run the cells sequentially to reproduce the results.

## 🤝 Contributing
Contributions are welcome! If you would like to contribute, please submit a pull request or raise an issue with your suggestions.

  
