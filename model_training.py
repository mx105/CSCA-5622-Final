"""
Yelp Review Sentiment Analysis
-----------------------------
This script performs sentiment analysis on Yelp reviews using natural language processing
and machine learning techniques. It processes raw review text, creates various n-gram models,
and compares the performance of different feature extraction methods.

The analysis pipeline includes:
1. Data preprocessing and cleaning
2. Text normalization (lemmatization)
3. Feature extraction using different n-gram models
4. Model training and evaluation
5. Visualization of results
"""

import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
def setup_nltk_resources():
    """Download required NLTK datasets."""
    resources = ['punkt', 'wordnet', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

# Dictionary of English contractions and their expansions
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

def clean_text(text: str, remove_stopwords: bool = True) -> list[str]:
    """Clean and preprocess text through multiple normalization steps.

    Processing pipeline:
    1. Convert to lowercase
    2. Expand contractions (e.g., "don't" -> "do not")
    3. Remove URLs, HTML elements, and special characters
    4. Optionally remove English stopwords
    5. Tokenize into words using punctuation-aware tokenization

    Args:
        text: Input text to be cleaned
        remove_stopwords: Whether to remove stopwords. Defaults to True.

    Returns:
        List of cleaned tokens

    Example:
        >>> clean_text("I'm testing http://example.com!", remove_stopwords=True)
        ['testing']
    """
    # Normalize case
    text = text.lower()

    # Expand contractions using lookup dictionary
    text = ' '.join([contractions.get(word, word) for word in text.split()])

    # Define regex substitutions with corresponding flags
    substitution_rules = [
        (r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', 0),  # Special characters
        (r'<br />', ' ', 0),                    # HTML line breaks potentially leftover from scraping
    ]

    # Apply all substitution rules
    for pattern, replacement, flags in substitution_rules:
        text = re.sub(pattern, replacement, text, flags=flags)

    # Remove stopwords if requested
    if remove_stopwords:
        words = text.split()
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in words if word not in stop_words])

    # Return punctuation-aware tokens
    return nltk.WordPunctTokenizer().tokenize(text)

def lemmatize_texts(texts):
    """
    Lemmatize a series of texts using WordNet lemmatizer.
    
    Args:
        texts (pd.Series): Series containing tokenized texts
    
    Returns:
        list: List of lemmatized texts
    """
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in text] for text in texts]

def create_wordclouds(df):
    """
    Generate and display word clouds for positive and negative reviews.
    
    Args:
        df (pd.DataFrame): DataFrame containing reviews and labels
    """
    # Combine reviews by sentiment
    negative_text = " ".join([" ".join(review) for review in df[df['Label'] == 0]['lemmatized_text']])
    positive_text = " ".join([" ".join(review) for review in df[df['Label'] == 1]['lemmatized_text']])
    
    # Create word clouds
    cloud_params = {'width': 800, 'height': 400, 'background_color': 'white'}
    negative_cloud = WordCloud(**cloud_params).generate(negative_text)
    positive_cloud = WordCloud(**cloud_params).generate(positive_text)
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(negative_cloud, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title("Negative Reviews Word Cloud")
    
    ax2.imshow(positive_cloud, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title("Positive Reviews Word Cloud")
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    # Setup
    setup_nltk_resources()
    
    # Load data
    df = pd.read_csv('Scraped_Data.csv')
    df.drop_duplicates(inplace=True)
    
    # Create binary labels (1 for positive reviews, 0 for negative)
    df['Label'] = (df['score'] > 3).astype(int)
    
    # Clean and process text
    df['Text_Cleaned'] = list(map(clean_text, df['review_text']))
    df['lemmatized_text'] = lemmatize_texts(df['Text_Cleaned'])
    
    # Generate visualizations
    create_wordclouds(df)
    
    # Prepare text data for modeling
    df['processed_text'] = df['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))
    X = df['processed_text']
    y = df['Label']
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )
    
    # Define pipeline with vectorizer and classifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],  # Test unigrams and bigrams
        'vectorizer__max_features': [500, 1000, 1500, 2000],     # Limit vocabulary size
        'vectorizer__max_df': [0.9, 1],  # Ignore overly common words
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], # Regularization strength
        'classifier__penalty': ['l1', 'l2']           # Regularization type
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Show best parameters and accuracy
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy: {:.2f}".format(grid_search.best_score_))
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nTest Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Get the best estimator from GridSearch
    best_model = grid_search.best_estimator_

    # Extract components
    vectorizer = best_model.named_steps['vectorizer']
    classifier = best_model.named_steps['classifier']

    # Get coefficients (for class 1)
    coefs = classifier.coef_[0]

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame of features and coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs,
        'abs_coef': np.abs(coefs)
    })

    # Filter out zero-impact features
    sparse_features = coef_df[coef_df['coef'] != 0]

    sparse_features_sorted = sparse_features.sort_values('abs_coef', ascending=False)

    # Top 10 positive-impact features (predict class 1)
    print("Features predicting POSITIVE reviews:")
    print(sparse_features_sorted[sparse_features_sorted['coef'] > 0].head(10))

    # Top 10 negative-impact features (predict class 0)
    print("\nFeatures predicting NEGATIVE reviews:")
    print(sparse_features_sorted[sparse_features_sorted['coef'] < 0].head(10))

    # Take top 20 features
    top_features = sparse_features_sorted.head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='coef', y='feature', data=top_features, palette='coolwarm')
    plt.title("Top Impactful Features (L1-Regularized Logistic Regression)")
    plt.xlabel("Coefficient Magnitude")
    plt.ylabel("Feature")
    plt.show()

if __name__ == "__main__":
    main()