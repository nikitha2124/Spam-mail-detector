# Spam Mail Detector

## Overview

Spam Mail Detector is a Machine Learning and Natural Language Processing (NLP) based web application that classifies emails and messages as **Spam** or **Ham (Not Spam)**. The project uses text preprocessing, TF-IDF vectorization, and machine learning algorithms to accurately identify unwanted messages.

## Features

* Real-time spam message detection
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Naive Bayes classification model
* Logistic Regression model comparison
* Confidence score prediction
* Interactive Streamlit web interface
* Model performance evaluation using Confusion Matrix

## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* NLTK
* Streamlit
* Joblib
* Matplotlib

## Project Structure

spam-mail-detector/

├── data/
│   └── spam.csv

├── models/
│   ├── spam_model.pkl
│   └── tfidf.pkl

├── results/
│   └── model_report.txt

├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── model_comparison.py

├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

## Machine Learning Workflow

1. Data Collection
2. Text Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training
5. Model Evaluation
6. Spam Prediction
7. Web Application Deployment

## Model Performance

* Naive Bayes Classifier
* Logistic Regression Classifier
* Accuracy Comparison
* Confusion Matrix Analysis

## Installation

Clone the repository:

git clone https://github.com/nikitha2124/Spam-mail-detector.git

Navigate to the project directory:

cd Spam-mail-detector

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Example

Input:
Congratulations! You have won a free iPhone. Click here to claim.

Output:
Spam

Input:
Hi, are we meeting tomorrow for the project discussion?

Output:
Ham

## Future Enhancements

* Deep Learning-based spam detection
* Email attachment analysis
* Multi-language spam detection
* Deployment on cloud platforms
* Improved accuracy with ensemble models

## Author

Nikitha

B.Tech – Data Science

Machine Learning & AI Enthusiast

