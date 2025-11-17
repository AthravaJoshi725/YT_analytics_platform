import logging
import config
import re
import time

from concurrent.futures import ThreadPoolExecutor

from services.models import load_sentiment_model, load_emotion_model, load_spam_model
from services.utils import preprocess_input
import pandas as pd

from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from collections import Counter
import nltk

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_PATHS['log_file']),
        logging.StreamHandler()
    ]
)

# load models once
sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()
spam_classifier = load_spam_model()

# label map for sentiment model
sentiment_label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

def analyze_sentiment(comments: list): 
    logging.info("Starting sentiment analysis...")

    try:
        start = time.time()
        result = []
        for c in comments:
            res = sentiment_model(c)[0]
            label = sentiment_label_map.get(res['label'], res['label'])
            result.append(label)

        end = time.time()

        logging.info(f'Sentiment analysis time take: {end -start:.2f}s')
        logging.info(f"Sentiment analysis completed for {len(comments)} comments.")
        
        return result

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return []



def analyze_emotion(comments: list):
    logging.info("Starting emotion analysis...")

    try:
        start = time.time()
        results = []
        for c in comments:
            res = emotion_model(c)[0]
            label = res['label']  # direct label, no mapping needed 
            results.append(label)
        end = time.time()

        logging.info(f"Emotion analysis completed for {len(comments)} comments.")
        logging.info(f'Emotion analysis time take: {end - start:.2f}s')
    except Exception as e:
        logging.error(f"Error during emotion analysis: {e}")

    return results

def analyze_spam(comments: list):
    '''
    scikit-learn model
    '''
    logging.info("Starting spam detection...")

    try:
        start = time.time()
        results = []
        for c in comments:
            preprocessed_c = preprocess_input(c)
            spam_label = spam_classifier.predict(preprocessed_c)[0]  
            results.append(int(spam_label))
        end = time.time()

        logging.info(f"Spam detection completed for {len(comments)} comments.")
        logging.info(f"Spam analysis time taken: {end - start:.2f}s")

    except Exception as e:
        logging.error(f"Error during spam detection: {e}")

    return results

def get_adjectives(comments: list):
    logging.info("Starting adjective extraction")
    adjectives = []
    try:
        start = time.time()

        # convert list in series
        comments = pd.Series(comments)
        # combine all text 
        text = " ".join(comments.astype(str))
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^A-Za-z\s]", "", text)
        text = text.lower()

        # tokenize + POS tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # stopwords and filter adjectives
        stop_words = set(stopwords.words("english"))
        adjectives = [word for word, tag in tagged if tag.startswith("JJ") and word not in stop_words]

        # optional: count top adjectives
        top_adjectives = Counter(adjectives).most_common(30)
        # print("Top adjectives:\n", top_adjectives)
        end = time.time()

        logging.info("Adjective extraction completed.")
        logging.info(f'Adjective extraction time take: {end - start:.2f}s')
    except Exception as e:
        logging.error(f"Error during adjective extraction: {e}")
    
    return adjectives

import nltk

def check_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
        logging.info("NLTK resources are downloaded")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def get_analysis(comments: list):

    try:
        start = time.time()
        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(analyze_sentiment,comments)
            f2 = executor.submit(analyze_emotion,comments)
            f3 = executor.submit(analyze_spam,comments)
            f4 = executor.submit(get_adjectives, comments)

            sentiment_label = f1.result()
            emotion_label = f2.result()
            spam_label = f3.result()
            adjectives = f4.result()
        
        end = time.time()
        logging.info(f"Total Analysis Time:{end - start:.2f}s ")

        return sentiment_label, emotion_label, spam_label, adjectives
        
    except Exception as e:
        logging.error(f"{e}")
        return None, None, None, None
    

if __name__ == "__main__":
    check_nltk_data()
    test_comments = [
        "I love this movie!",
        "This was terrible, I hated it.",
        "It’s fine, nothing special.",
        "Congratulations! You've won a free ticket. Click here to claim."
    ]
    sentiment_result, emotion_result, spam_result, adjectives = get_analysis(test_comments)

    print("\n===  Analysis Test ===\n")
    print("Sentiment:", sentiment_result)
    print("Emotoin:", emotion_result)
    print('spam:',spam_result)
    print("Adjectives:", adjectives)