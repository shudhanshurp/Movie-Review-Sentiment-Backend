from flask import request, jsonify, Flask
from flask_cors import CORS
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


app = Flask(__name__)
CORS(app)


def remove_special_character(content):
    # re.sub('\[[^&@#!]]*\]', '', content) '\W+' matches one or more non-word characters (e.g., punctuation, symbols)
    return re.sub('\W+', ' ', content)

# Removing URL's


def remove_url(content):
    return re.sub(r'http\S+', '', content)

# Removing the stopwords from text


def remove_stopwords(content):
    stop_words = stopwords.words('english')
    print
    new_stopwords = ["would", "shall", "could", "might"]
    stop_words.extend(new_stopwords)
    # we specifically want this in the text corpus as it will help to understand the emotion of the reviewer
    stop_words.remove("not")
    stop_words = set(stop_words)
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)

# Expansion of english contractions


def contraction_expansion(content):
    content = contractions.fix(content)
    return content

# Data preprocessing


def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)

    content = remove_stopwords(content)
    return content


def get_tokenizer():
    global tfidfvect
    with open('tfidfvect.pkl', 'rb') as file:
        tfidfvect = pickle.load(file)
    print("Vectorizer Loaded!")


def get_model():
    global model

    with open('IMDB_review_predictor.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model Loaded!")


print("Loading model...")
get_model()
get_tokenizer()


@app.route("/")
def running():
    return "Flask is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.get_json(force=True)
        user_input_encoded = data_cleaning(message['text'])
        user_input_encoded = [user_input_encoded]
        user_input_transformed = tfidfvect.transform(user_input_encoded)

        user_input_dense = user_input_transformed.toarray()

        predicted_prob = model.predict(user_input_dense)

        print(predicted_prob)

        response = {
            'sentiment': str(predicted_prob),
        }

        return jsonify(response)
    except Exception as e:
        print(str(e) + " this is fuck up")
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run()
