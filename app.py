from flask import Flask, render_template, request, flash, redirect, url_for
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import matplotlib
matplotlib.use('Agg') 

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

# Tokenization function
def tokenize_text(text):
    return sent_tokenize(text), word_tokenize(text)

# Sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Named Entity Recognition function
def extract_entities(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    entities = ne_chunk(tagged)
    return {
        'Persons': [ ' '.join(c[0] for c in chunk) for chunk in entities if isinstance(chunk, Tree) and chunk.label() == 'PERSON'],
        'Locations': [ ' '.join(c[0] for c in chunk) for chunk in entities if isinstance(chunk, Tree) and chunk.label() == 'GPE'],
        'Organizations': [ ' '.join(c[0] for c in chunk) for chunk in entities if isinstance(chunk, Tree) and chunk.label() == 'ORGANIZATION']
    }

# Advanced Topic Modeling function using TF-IDF with LDA
def perform_topic_modeling(text, num_topics=5):
    # Step 1: Convert text into a document-term matrix using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    
    # Step 2: Apply LDA on the TF-IDF matrix
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)

    # Step 3: Extract the topics
    topics = []
    for topic in lda.components_:
        topic_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics.append(topic_words)
    
    return topics

# Create sentiment plot function
def create_sentiment_plot(sentiment_scores):
    labels = ['Negative', 'Neutral', 'Positive']
    values = [sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos']]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text='Sentiment Analysis')
    return fig.to_html(full_html=False)

# Generate word cloud function
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f'static/wordcloud_{title}.png')  # Save to static folder
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    file = request.files.get('file')

    if not text and not file:
        flash("No file or text provided. Please enter text or upload a .txt file.", "error")
        return redirect(url_for('index'))

    if file:
        if not file.filename.endswith('.txt'):
            flash("Invalid file format. Please upload a .txt file.", "error")
            return redirect(url_for('index'))
        text = file.read().decode('utf-8')

    if not text.strip():
        flash("Text is empty. Please enter some text.", "error")
        return redirect(url_for('index'))

    sentences, words = tokenize_text(text)
    sentiment_scores = analyze_sentiment(text)
    topics = perform_topic_modeling(text)
    sentiment_plot = create_sentiment_plot(sentiment_scores)
    
    # Generate and save word clouds for the example topics
    for idx, topic in enumerate(topics):
        topic_text = ' '.join(topic)
        generate_word_cloud(topic_text, f'Topic_{idx+1}')

    results = {
        'num_sentences': len(sentences),
        'first_sentence': sentences[0] if sentences else "No sentences found.",
        'sentiment_scores': sentiment_scores,
        'topics': topics,
        'entities': extract_entities(text),
        'sentiment_plot': sentiment_plot,
        'wordcloud_images': [f'wordcloud_Topic_{idx+1}.png' for idx in range(len(topics))]
    }
    
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
