import pandas as pd
from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean messages
df['message'] = df['message'].apply(clean_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['message']).toarray()
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Naive Bayes Model
# ------------------------
nb = MultinomialNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, nb_pred)

print("\nNaive Bayes Confusion Matrix:")
print(cm)

# Accuracy
nb_acc = accuracy_score(y_test, nb_pred)

# ------------------------
# Logistic Regression Model
# ------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)

# ------------------------
# Results
# ------------------------
print("\nModel Comparison")
print("-" * 30)

print(f"Naive Bayes Accuracy          : {nb_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy : {lr_acc * 100:.2f}%")

if nb_acc > lr_acc:
    print("\nBest Model: Naive Bayes")
else:
    print("\nBest Model: Logistic Regression")

# Sample Predictions
print("\nSample Predictions using Naive Bayes")

sample1 = ["Congratulations! You have won a free iPhone. Claim now."]
sample1_vec = tfidf.transform(sample1)

result1 = nb.predict(sample1_vec)[0]
print("Message 1:", "Spam" if result1 == 1 else "Ham")

sample2 = ["Hi, are we meeting tomorrow at 10 AM?"]
sample2_vec = tfidf.transform(sample2)

result2 = nb.predict(sample2_vec)[0]
print("Message 2:", "Spam" if result2 == 1 else "Ham")