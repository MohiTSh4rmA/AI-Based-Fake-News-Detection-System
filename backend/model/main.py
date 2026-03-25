from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
from nltk.corpus import stopwords

app = FastAPI()

# Enable CORS (React connection ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev mode only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = pickle.load(open("model/saved_model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/saved_model/vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

# Request body schema
class NewsRequest(BaseModel):
    news: str

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@app.get("/")
def home():
    return {"message": "Fake News Detection API Running 🚀"}

@app.post("/predict")
def predict(request: NewsRequest):
    news = request.news
    cleaned = clean_text(news)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0].max()

    result = "REAL ✅" if prediction == 1 else "FAKE ❌"

    return {
        "prediction": result,
        "confidence_percentage": round(probability * 100, 2)
    }