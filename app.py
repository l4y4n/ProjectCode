
#.\tf_env\Scripts\python.exe app.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import re
import joblib
import pickle
import numpy as np
import threading
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyarabic.araby import strip_tashkeel, normalize_hamza
from qalsadi.lemmatizer import Lemmatizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import sys
import tensorflow as tf
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from imblearn.over_sampling import RandomOverSampler
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras


def arabic_tokenizer(text):
    text = normalize_hamza(strip_tashkeel(text))
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    return [t for t in text.split() if len(t) > 2 and t not in stopwords.words('arabic')]

# Make it available for unpickling
sys.modules['__main__'].arabic_tokenizer = arabic_tokenizer

class KerasClassifierWrapper(ClassifierMixin):
    def __init__(self, model, tokenizer, max_len=100):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Clear optimizer warnings
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def predict(self, X):
        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')
        return (self.model.predict(padded) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')
        # Ensure output shape matches sklearn (n_samples, n_classes)
        proba = self.model.predict(padded)
        return np.column_stack((1-proba, proba))  # Shape (n_samples, 2)

    def fit(self, X, y=None):
        return self  # Already pretrained

class TfidfVectorizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer_path):
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None

    def fit(self, X, y=None):
        self.vectorizer = joblib.load(self.vectorizer_path)
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Loaded vectorizer is not fitted!")
        return self

    def transform(self, X):
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not loaded. Call fit() first.")
        return self.vectorizer.transform(X)

try:
    # First try loading with fixed tokenizer
    sys.modules['__main__'].arabic_tokenizer = arabic_tokenizer
    ensemble = joblib.load(r'C:\\Users\\Lenovo\\Desktop\\idk\\full_ensemble.pkl')
except Exception as e:
    print(f"Model load failed: {e}")
    try:
        # Fallback to manual component loading
        from sklearn.ensemble import VotingClassifier
        logreg = joblib.load(r'C:\\Users\\Lenovo\\Desktop\\idk\\Logical_Regresion.pkl')
        svm = joblib.load(r'C:\\Users\\Lenovo\\Desktop\\idk\\svm_model.pkl') 
        rf = joblib.load(r'C:\\Users\\Lenovo\\Desktop\\idk\\random_forest_model.pkl')
        ensemble = VotingClassifier([
            ('lr', logreg),
            ('svm', svm),
            ('rf', rf)
        ], voting='soft')
        print("Loaded individual models successfully")
    except Exception as e:
        print(f"Critical failure: {e}")
        ensemble = None


app = Flask(__name__)
CORS(app)

# Download stopwords/tokenizer
nltk.download('stopwords')
nltk.download('punkt')

try:
    ensemble = joblib.load('full_ensemble.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    ensemble = None

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            sentence = data.get('sentence')

            if not sentence:
                return jsonify({'error': 'No sentence provided'}), 400
            if not is_arabic(sentence):
                return jsonify({'error': 'Sentence must be in Arabic'}), 400
            if not ensemble:
                return jsonify({"error": "Model not loaded"}), 500

            cleaned_text = clean_arabic_text(sentence)
    

                
            prediction = ensemble.predict([cleaned_text])[0]
            return jsonify({"prediction": "مكتئب" if prediction == 1 else "غير مكتئب"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
thread_local = threading.local()

def get_lemmatizer():
    """
    Get or create a Lemmatizer instance for the current thread.
    """
    if not hasattr(thread_local, "lemmatizer"):
        thread_local.lemmatizer = Lemmatizer()
    return thread_local.lemmatizer

# Cache lemmatization results to avoid repeated SQLite access
@lru_cache(maxsize=1000)  # Cache up to 1000 lemmatized words
def lemmatize_word(word):
    lemmatizer = get_lemmatizer()  # Get the thread-local lemmatizer
    return lemmatizer.lemmatize(word)


# Text cleaning function
def is_arabic(text):
    """
    Check if the text is Arabic.
    """
    return bool(re.search('[\u0600-\u06FF]', text))

def clean_arabic_text(text):
    """
    Clean and preprocess Arabic text.
    """
    if is_arabic(text):
        text = strip_tashkeel(text)  # Remove diacritics
        text = normalize_hamza(text)  # Normalize hamza
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = ' '.join(text.split()).lower()  # Normalize whitespace and lowercase
        stop_words = set(stopwords.words('arabic'))  # Arabic stopwords
        words = word_tokenize(text)  # Tokenize the text
        filtered_text = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(lemmatize_word(word) for word in filtered_text)  # Apply lemmatization
    return text


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)  # Run in single-threaded mode
