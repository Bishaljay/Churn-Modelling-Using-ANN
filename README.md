# Churn Modelling Using ANN

This project implements an **Artificial Neural Network (ANN)** to predict customer churn based on various demographic, geographic, and account-related factors. 

The goal is to classify whether a customer will **leave (churn)** or **stay** with the service provider.

---

## 📌 Project Overview
- **Objective**: Predict customer churn using a deep learning model (ANN).
- **Dataset**: Typically includes features like:
  - Customer demographics (e.g., gender, age, geography)
  - Account information (e.g., balance, number of products, active status)
  - Credit score and tenure
- **Model Type**: Artificial Neural Network (ANN) built using **Keras/TensorFlow**.
- **Output**: Binary classification — churn (`1`) or not churn (`0`).

---

## ⚙️ Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
````

---

## 📂 Project Structure

```
├── Churn Modelling Using ANN.ipynb   # Main Jupyter Notebook
├── dataset.csv                       # Dataset file
└── README.md                         # Project documentation
```

## 🧠 Model Workflow

1. **Data Preprocessing**

   * Load dataset
   * Encode categorical variables
   * Scale numerical features
   * Split into training and test sets

2. **Model Building**

   * Input layer with feature size
   * One or more hidden layers (ReLU activation)
   * Output layer (Sigmoid activation for binary classification)

3. **Model Training**

   * Loss: `binary_crossentropy`
   * Optimizer: `adam`
   * Metrics: `accuracy`

4. **Evaluation**

   * Model performance on the test set
   * Confusion matrix and classification report

---

## 📊 Results

* Achieved validation accuracy of about 85.73% and a training accuracy around 83.99%.
* Visualized learning curves for **loss** and **accuracy**.
* Model can predict churn probability for new customers.

---

## 📌 Future Improvements

* Hyperparameter tuning (layers, neurons, learning rate)
* Trying other models (Random Forest, Gradient Boosting)
* Feature engineering for better accuracy

---
## 📜 License

This project is for educational purposes.
