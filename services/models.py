from transformers import pipeline
import os
import sys
import pickle as pkl
import config

def load_sentiment_model():
    local_path = config.MODELS_DIR / "sentiment_model"
    model = pipeline("sentiment-analysis", model=str(local_path), truncation=True, max_length=512)

    return model

def load_emotion_model():
    local_path = config.MODELS_DIR / "emotion_model"
    # folder = [os.path.join(local_path, f) for f in os.listdir(local_path)][0]
    model = pipeline("text-classification", model=str(local_path), max_length=512, truncation=True)
    # takes first two scores
    return model

def load_spam_model():
    model_path = config.MODELS_DIR / "spam_detection" / "spam_classifier_model.pkl"
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
        return model

if __name__ == "__main__":
    # model = load_sentiment_model()
    emotion_model = load_emotion_model()

    # sample comments
    test_comments = [
        "This movie was amazing!",
        "Worst acting ever, waste of time.",
        "It was okay, nothing special."
    ]

    results = load_sentiment_model(test_comments)

    # print nicely
    for comment, res in zip(test_comments, results):
        print(f"Comment: {comment}")
        print(f"Label: {res['label']}, Score: {round(res['score'], 3)}")
        print("-" * 40)
