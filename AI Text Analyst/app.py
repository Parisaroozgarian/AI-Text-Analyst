from flask import Flask, render_template, request
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)

# Tokenization function
def tokenize_text(text):
    return sent_tokenize(text), word_tokenize(text)

# Sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Topic modeling function
def perform_topic_modeling(text, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([text])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    return [
        [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
        for topic in lda.components_
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    file = request.files.get('file')

    if file and file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    elif not text:
        return "No file or text provided", 400

    # Perform analysis
    sentences, words = tokenize_text(text)
    sentiment_scores = analyze_sentiment(text)
    topics = perform_topic_modeling(text)

    results = {
        'num_sentences': len(sentences),
        'first_sentence': sentences[0] if sentences else "No sentences found.",
        'sentiment_scores': sentiment_scores,
        'topics': topics,
    }
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
