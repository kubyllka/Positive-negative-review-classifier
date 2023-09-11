# Import all necessary libraries
import nltk
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import string

nltk.download('punkt')
nltk.download('stopwords')


# Define a function for text preprocessing
def preprocess_text(text):
    """
    Takes an input text, converts it to lowercase,
    tokenizes it,
    removes punctuation and common English stop words,
    applies stemming to the remaining tokens,
    and returns the preprocessed text as a string.

    Args:
        text (string): text that must be pre-processed

    Returns:
        the preprocessed text as a string
    """

    text = text.lower()
    tokens = word_tokenize( text )

    # Stop_words in English like "i", "me", "we", "myself" and many others
    stop_words = set(stopwords.words('english'))

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stop_words
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Remove morphological affixes from words, leaving only the word stem
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Return stemmed words
    return " ".join(stemmed_tokens)

# Create an argument parser
parser = argparse.ArgumentParser(description='Positive or negative review detector')

# Define the input file argument
parser.add_argument('input_file', type=str, help='Path to the input CSV file (e.g., test_reviews.csv)')

# Define an optional output file argument
parser.add_argument( '--output_file', type=str, default='test_labels_pred.csv', help='Path to the output CSV file')

# Parse the command-line arguments
args = parser.parse_args()

# Load the trained model
model = load_model('model_conv.h5')

# Load the test data from the input CSV file
test_data = pd.read_csv(args.input_file)

# Define parameters for tokenization
vocab_size = 1000
oov_tok = "<OOV>"
max_length = 120
trunc_type='post'

# Preprocess the text data in the 'text' column
test_data['text'] = test_data['text'].apply(preprocess_text)

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(test_data['text'])

sequences = tokenizer.texts_to_sequences(test_data['text'])
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

# Perform inference on the preprocessed and padded test data
predictions = model.predict(padded)

# Convert predictions to binary labels (0 or 1)
rounded_predictions = np.round(predictions).astype(int)
rounded_predictions = rounded_predictions.flatten()

# Create a DataFrame with Review ID and Predicted Label
results_df = pd.DataFrame({'id': test_data['id'], 'sentiment': rounded_predictions})

# Map the numeric labels to 'positive' and 'negative'
results_df['sentiment'] = results_df['sentiment'].map({0: 'Negative', 1: 'Positive'})

# Save the predictions to the specified output file or the default 'test_labels_pred.csv'
results_df.to_csv(args.output_file, index=False)







