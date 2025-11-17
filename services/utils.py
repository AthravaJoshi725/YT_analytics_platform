import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle as pkl
def preprocess_input(text):
    # dump transformer and vectorize
    count_vect = pkl.load(open('models/count_vectorizer.pkl','rb'))
    transformer = pkl.load(open('models/TfidfTransformer.pkl','rb'))
    # 


    text = " ".join(re.findall(r'[a-zA-Z]', text.lower()))
    input_counts = count_vect.transform([text])
    input_tfidf = transformer.transform(input_counts)
    return input_tfidf


# testing
if __name__ == "__main__":
    sample_text = "This is a sample"
    processed = preprocess_input(sample_text)
    print(processed)