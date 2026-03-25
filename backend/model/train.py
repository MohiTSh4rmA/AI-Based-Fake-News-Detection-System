import pandas as pd
import re
import nltk
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load datasets
fake = pd.read_csv("../../dataset/Fake.csv")
true = pd.read_csv("../../dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Use only 10000 samples for fast training
data = data.head(10000)

data["text"] = data["title"] + " " + data["text"]
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

print("\nModel Performance:\n")
print(classification_report(y_test, model.predict(X_test)))

# Save model
os.makedirs("saved_model", exist_ok=True)
pickle.dump(model, open("saved_model/model.pkl", "wb"))
pickle.dump(vectorizer, open("saved_model/vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")