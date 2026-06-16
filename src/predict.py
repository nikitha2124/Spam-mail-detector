import pickle
from src.preprocess import clean_text

model = pickle.load(open("models/spam_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

def predict_message(msg):
    msg = clean_text(msg)

    vec = tfidf.transform([msg]).toarray()

    prediction = model.predict(vec)[0]

    confidence = max(model.predict_proba(vec)[0])

    return prediction, confidence