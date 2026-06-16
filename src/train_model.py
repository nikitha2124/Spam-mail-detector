import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from src.preprocess import clean_text

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocess text
df['message'] = df['message'].apply(clean_text)

# Features & labels
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['message']).toarray()
y = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/spam_model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))

print("Model trained and saved successfully!")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy = accuracy_score(y_test, y_pred)

with open("results/model_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n\n")
    f.write(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Report saved to results/model_report.txt")