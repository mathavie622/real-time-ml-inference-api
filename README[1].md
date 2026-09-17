
# Real-Time ML Inference REST API

## Project Overview

This project implements a real-time Machine Learning inference REST API using FastAPI.

A trained PyTorch sentiment classification model is packaged into a REST API that accepts text input and returns the predicted sentiment along with prediction probability and confidence.

The application is designed as a lightweight ML microservice and can be containerized using Docker.

---

## Problem Statement

Machine Learning models are useful only when they can be integrated into real-world applications.

This project solves the problem of serving a trained NLP model through a REST API so that external applications can send text and receive predictions in real time.

---

## Objectives

- Build a real-time ML inference API.
- Create a `/predict` REST endpoint.
- Accept JSON text input.
- Return sentiment prediction and probability.
- Save and load the trained PyTorch model.
- Test the API using automated unit tests.
- Dockerize the ML service.
- Document the complete ML system.

---

## Machine Learning Model

The project uses a text classification model built with PyTorch.

### Processing Pipeline

```text
Input Text
    ↓
TF-IDF Vectorization
    ↓
PyTorch Neural Network
    ↓
Softmax Probability
    ↓
Sentiment Prediction
    ↓
REST API Response
