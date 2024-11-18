import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models
nltk.download('stopwords')  # Download stopwords list

from flask import Flask, request, jsonify, render_template
from nltk.tokenize import sent_tokenize
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Initialize Flask app
app = Flask(__name__)

# Function to summarize text
def summarize_text(text, num_sentences=3):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords, and count word frequencies
    word_frequencies = {}
    for word in words:
        if word not in stop_words and word not in string.punctuation:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Normalize word frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    # Tokenize sentences and score them based on word frequencies
    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    # Select the top N sentences
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = " ".join(summary_sentences)
    return summary

# Route for the home page (serving index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for text summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Please provide 'text' in the request body"}), 400
    
    text = data['text']
    num_sentences = data.get('num_sentences', 3)  # Default to 3 sentences if not provided
    summary = summarize_text(text, num_sentences=num_sentences)
    return jsonify({"summary": summary})

# Run the Flask app
if __name__ == "__main__":
    # Ensure the "templates" folder exists for serving HTML files
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Place your index.html file in the 'templates' directory
    app.run(debug=True)
