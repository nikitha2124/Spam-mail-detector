# 📧 Spam Mail Detector

A Machine Learning and Natural Language Processing (NLP) based web application that detects whether a message is **Spam** or **Ham (Not Spam)**. The project uses TF-IDF vectorization and classification algorithms to analyze text and provide accurate predictions through an interactive Streamlit interface.

---

## 🚀 Project Overview

Spam messages are a common problem in email and messaging systems. This project aims to automatically classify incoming messages into:

* **Spam** ❌
* **Ham (Legitimate Message)** ✅

The application preprocesses text, extracts meaningful features using TF-IDF, and applies Machine Learning algorithms to make predictions.

---

## 🎯 Objectives

* Detect spam messages with high accuracy.
* Apply Natural Language Processing techniques.
* Compare multiple Machine Learning algorithms.
* Provide real-time predictions through a user-friendly web interface.
* Evaluate model performance using standard metrics.

---

## ✨ Features

* Real-time Spam/Ham classification
* Text cleaning and preprocessing
* TF-IDF feature extraction
* Naive Bayes classifier
* Logistic Regression comparison
* Confidence score prediction
* Confusion Matrix evaluation
* Interactive Streamlit web application
* Easy deployment and scalability

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib
* Matplotlib

### Machine Learning Techniques

* TF-IDF Vectorization
* Multinomial Naive Bayes
* Logistic Regression
* Text Classification

---

## 📂 Project Structure

```text
spam-mail-detector/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf.pkl
│
├── results/
│   └── model_report.txt
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── model_comparison.py
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 Workflow

### 1. Data Collection

The dataset contains labeled messages categorized as spam or ham.

### 2. Data Preprocessing

* Convert text to lowercase
* Remove punctuation
* Remove stopwords
* Tokenization
* Text cleaning

### 3. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features.

### 4. Model Training

The dataset is trained using:

* Multinomial Naive Bayes
* Logistic Regression

### 5. Model Evaluation

Performance is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### 6. Prediction

The trained model predicts whether a given message is spam or not.

### 7. Deployment

The model is deployed using Streamlit for interactive usage.

---

## 📊 Model Performance

Models evaluated:

### Naive Bayes

* Fast training
* Effective for text classification
* High accuracy on spam detection tasks

### Logistic Regression

* Strong baseline classifier
* Good interpretability
* Used for comparison

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/nikitha2124/Spam-mail-detector.git
```

### Navigate to Project

```bash
cd Spam-mail-detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🧪 Sample Predictions

### Example 1

Input:

Congratulations! You have won a free iPhone. Click here to claim your reward.

Output:

Spam ❌

### Example 2

Input:

Hi Nikitha, are we meeting tomorrow for the project discussion?

Output:

Ham ✅

---

## 🔮 Future Enhancements

* Deep Learning based classification
* LSTM/BERT implementation
* Email attachment analysis
* Multi-language spam detection
* Cloud deployment
* Improved ensemble models
* Real-time email integration

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Natural Language Processing
* Machine Learning Model Development
* Feature Engineering
* Model Evaluation
* Streamlit Application Development
* Git & GitHub Version Control
* Project Deployment

---

## 👩‍💻 Author

**Nikitha**

B.Tech – Data Science

Machine Learning | Artificial Intelligence | Data Analytics Enthusiast

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.

