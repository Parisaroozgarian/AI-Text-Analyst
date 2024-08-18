from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import io
import base64
from collections import Counter
import pandas as pd
from fpdf import FPDF

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Tokenization function
def tokenize_text(text):
    return sent_tokenize(text), word_tokenize(text)

# Sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Topic modeling function
def perform_topic_modeling(text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([text])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    return [
        [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-3:]]
        for topic in lda.components_
    ]

# Named Entity Recognition function
def extract_entities(text):
    doc = nlp(text)
    
    # Initialize dictionaries to store entities
    entities = {
        'persons': set(),
        'locations': set(),
        'organizations': set(),
        'others': set()
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities['persons'].add(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities['locations'].add(ent.text)
        elif ent.label_ == "ORG":
            entities['organizations'].add(ent.text)
        else:
            entities['others'].add(f"{ent.text}: {ent.label_}")
    
    # Convert sets to lists for HTML rendering
    return {key: list(value) for key, value in entities.items()}

# Sentiment plot function
def plot_sentiment(sentiment_scores):
    fig = px.bar(
        x=list(sentiment_scores.keys()),
        y=list(sentiment_scores.values()),
        labels={'x': 'Sentiment', 'y': 'Score'},
        title='Sentiment Analysis',
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_html(full_html=False)

# Topic distribution plot function
def plot_topic_distribution(topics):
    topic_labels = [', '.join(topic) for topic in topics]
    topic_counts = [1] * len(topic_labels)

    fig = go.Figure(data=[go.Pie(
        labels=topic_labels,
        values=topic_counts,
        hole=0.3,
        marker=dict(colors=['gold', 'lightgreen', 'lightcoral']),
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title='Topic Distribution',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig.to_html(full_html=False)

# Named entity frequency plot function
def plot_entity_frequency(entities):
    # Flatten the list of entities
    all_entities = [ent for sublist in entities.values() for ent in sublist]
    entity_freq = {ent: all_entities.count(ent) for ent in set(all_entities)}
    
    fig = go.Figure(data=[go.Bar(
        x=list(entity_freq.keys()),
        y=list(entity_freq.values()),
        marker_color='lightblue'
    )])
    
    fig.update_layout(
        title='Named Entity Frequency',
        xaxis_title='Entity',
        yaxis_title='Frequency',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig.to_html(full_html=False)

# Word Cloud plot function
def plot_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Convert word cloud to an image that can be rendered in HTML
    buf = io.BytesIO()
    wordcloud.to_image().save(buf, format='PNG')
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f'<img src="data:image/png;base64,{img_str}"/>'

# Word frequency plot function
def plot_word_frequency(text):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha()]
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)

    fig = px.bar(
        x=[word for word, _ in most_common_words],
        y=[freq for _, freq in most_common_words],
        labels={'x': 'Word', 'y': 'Frequency'},
        title='Most Frequent Words'
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_html(full_html=False)

# Text statistics function
def text_statistics(text):
    words = word_tokenize(text)
    num_words = len([word for word in words if word.isalpha()])
    num_sentences = len(sent_tokenize(text))
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    return {
        'num_words': num_words,
        'num_sentences': num_sentences,
        'avg_sentence_length': avg_sentence_length
    }

# Flask routes
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
        flash("Please provide text or upload a .txt file.", "error")
        return redirect(url_for('index'))

    # Perform analysis
    sentences, _ = tokenize_text(text)
    sentiment_scores = analyze_sentiment(text)
    topics = perform_topic_modeling(text)
    entities = extract_entities(text)
    sentiment_plot_html = plot_sentiment(sentiment_scores)
    topic_distribution_html = plot_topic_distribution(topics)
    entity_frequency_html = plot_entity_frequency(entities)
    word_cloud_html = plot_word_cloud(text)
    word_freq_html = plot_word_frequency(text)
    stats = text_statistics(text)

    # Limit the number of entities displayed
    max_entities = 10  # Maximum number of entities to display per type
    if isinstance(entities, dict):
        limited_entities = {key: value[:max_entities] for key, value in entities.items()}
    else:
        limited_entities = {}

    results = {
        'num_sentences': len(sentences),
        'first_sentence': sentences[0] if sentences else '',
        'sentiment_scores': sentiment_scores,
        'topics': topics,
        'entities': limited_entities,
        'sentiment_plot': sentiment_plot_html,
        'topic_distribution': topic_distribution_html,
        'entity_frequency': entity_frequency_html,
        'word_cloud': word_cloud_html,
        'word_frequency': word_freq_html,
        'text_statistics': stats
    }

    return render_template('results.html', results=results)

@app.route('/compare', methods=['POST'])
def compare():
    text1 = request.form.get('text1')
    text2 = request.form.get('text2')

    if not (text1 and text2):
        flash("Please provide two texts for comparison.", "error")
        return redirect(url_for('index'))

    # Perform analysis on both texts
    stats1 = text_statistics(text1)
    stats2 = text_statistics(text2)
    
    return render_template('compare_results.html', stats1=stats1, stats2=stats2)

@app.route('/download/<file_type>')
def download(file_type):
    text = request.args.get('text')
    
    if file_type == 'csv':
        df = pd.DataFrame([text_statistics(text)])
        df.to_csv('results.csv', index=False)
        return send_file('results.csv', as_attachment=True)
    elif file_type == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(text_statistics(text)))
        pdf.output("results.pdf")
        return send_file('results.pdf', as_attachment=True)
    else:
        flash("Invalid file type requested.", "error")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
