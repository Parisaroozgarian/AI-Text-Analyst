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
  
  <img width="1279" alt="Screen Shot 2024-08-22 at 12 59 06 AM" src="https://github.com/user-attachments/assets/bab01c44-621a-47b9-970d-7aab670b2f78">


- **Analysis Results**:
  
  <img width="1279" alt="Screen Shot 2024-08-22 at 1 28 12 AM" src="https://github.com/user-attachments/assets/535eb7bf-dedb-412d-bb34-86cf3d388519">


## Usage Examples 📋

### Example 1: Sentiment Analysis

**Input**: "I love this app! It's fantastic."

**Output**: 
- Sentiment: Positive
- Sentiment Plot:
  
  <img width="480" alt="Screen Shot 2024-08-22 at 1 30 13 AM" src="https://github.com/user-attachments/assets/8c540c7a-9e01-4093-aa0c-d4aa514d80c3">


### Example 2: Topic Modeling

**Input**: "Text data is crucial for machine learning. Analyzing text can reveal patterns."

**Output**:
- Topics: 
  1. Machine Learning, Data, Patterns
  2. Text Analysis, Revealing, Data

## Getting Started 🚀

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd <project-directory>
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Application**:
   Open your web browser and go to [http://localhost:5000](http://localhost:5000).
