# Text Analysis Application 📊

## Project Overview

The Text Analysis Application is an interactive web tool designed to perform comprehensive analysis on textual data. Users can input text directly or upload `.txt` files for analysis. The application leverages natural language processing (NLP) techniques to provide insights into sentiment, key topics, and named entities within the text, and visualizes these insights through interactive plots and word clouds.

## Key Features 🚀

- **Text Input and File Upload**: Users can either paste text into a text area or upload a `.txt` file for analysis.
- **Sentiment Analysis**: Evaluates the sentiment of the text and provides a visual representation of sentiment distribution (positive, neutral, negative).
- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to extract and display the main topics discussed in the text.
- **Named Entity Recognition**: Identifies and categorizes named entities such as people, locations, and organizations.
- **Word Clouds**: Generates visual representations of significant words in the topics for an intuitive understanding of key terms.

## Libraries & Technologies 🛠️

- **Flask**: A lightweight web framework used to build and run the application server.
- **NLTK (Natural Language Toolkit)**: Provides tools for text tokenization, sentiment analysis, and named entity recognition.
- **Scikit-Learn**: Implements machine learning algorithms for topic modeling using TF-IDF and Latent Dirichlet Allocation (LDA).
- **Plotly**: Generates interactive visualizations, including pie charts for sentiment analysis.
- **WordCloud**: Creates word cloud images to visually represent frequent terms in the topics.
- **Matplotlib**: Used for rendering and saving word cloud images.

## Screenshots 📸

Here are some screenshots of the application in action:

- **Home Page**:
  ![Home Page](static/images/home_page.png)

- **Analysis Results**:
  ![Analysis Results](static/images/results_page.png)

## Usage Examples 📋

### Example 1: Sentiment Analysis

**Input**: "I love this app! It's fantastic."

**Output**: 
- Sentiment: Positive
- Sentiment Plot: ![Sentiment Plot](static/sentiment_plot.png)

### Example 2: Topic Modeling

**Input**: "Text data is crucial for machine learning. Analyzing text can reveal patterns."

**Output**:
- Topics: 
  1. Machine Learning, Data, Patterns
  2. Text Analysis, Revealing, Data

## Getting Started 🚀

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pariasaroozgarian/AI-Text-Analyst.git
Navigate to the Project Directory:
bash
Copy code
cd AI-Text-Analyst
Install Dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Application:
bash
Copy code
python app.py
Access the Application:
Open your web browser and go to http://localhost:5000.