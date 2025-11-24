from transformers import pipeline
import os
import sys
import pickle as pkl

def load_sentiment_model():
    local_path = r"C:\Users\Admin\.cache\huggingface\hub\models--cardiffnlp--twitter-roberta-base-sentiment\snapshots"
    
    folder = [os.path.join(local_path, f) for f in os.listdir(local_path)][0]
    model = pipeline("sentiment-analysis", model=folder, truncation=True, max_length=512)

    return model

def load_emotion_model():
    local_path = r"C:\Users\Admin\.cache\huggingface\hub\models--j-hartmann--emotion-english-distilroberta-base\snapshots\manual_download"
    # folder = [os.path.join(local_path, f) for f in os.listdir(local_path)][0]
    model = pipeline("text-classification", model=local_path, max_length=512, truncation=True)
    # takes first two scores
    return model

def load_spam_model():
    model_path = 'models\spam_classifier_model.pkl'
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

    results = emotion_model(test_comments)

    # print nicely
    for comment, res in zip(test_comments, results):
        print(f"Comment: {comment}")
        print(f"Label: {res['label']}, Score: {round(res['score'], 3)}")
        print("-" * 40)
