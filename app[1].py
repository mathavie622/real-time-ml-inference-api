
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")


# Model architecture
class SentimentClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)


# Load model
input_size = len(vectorizer.vocabulary_)

model = SentimentClassifier(input_size)
model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)
model.eval()


# Create FastAPI app
app = FastAPI(
    title="Real-Time ML Inference API",
    description="Sentiment classification API using PyTorch and FastAPI",
    version="1.0.0"
)


# Request schema
class PredictionRequest(BaseModel):
    text: str


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert text to TF-IDF
    text_vector = vectorizer.transform([request.text]).toarray()

    input_tensor = torch.tensor(
        text_vector,
        dtype=torch.float32
    )

    # Model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    positive_probability = probabilities[0][1].item()
    negative_probability = probabilities[0][0].item()

    if positive_probability >= 0.5:
        prediction = "positive"
        probability = positive_probability
    else:
        prediction = "negative"
        probability = negative_probability

    return {
        "text": request.text,
        "prediction": prediction,
        "probability": round(probability, 4),
        "confidence": round(probability, 4)
    }


@app.get("/")
def home():
    return {
        "message": "Real-Time ML Inference API is running"
    }
