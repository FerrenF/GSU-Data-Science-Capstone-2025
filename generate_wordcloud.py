import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import json

# Load the dataset
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    return df

# Get negative sentament words(threshold determines how negative the text is we accept)
def get_strong_negative_words(reviews, threshold=-0.3):
    negative_words = []
    for review in reviews:
        blob = TextBlob(review)
        if blob.sentiment.polarity < threshold:
            for word, pos in blob.tags:
                if pos in ['JJ', 'VB', 'RB']:
                    word_lower = word.lower()
                    if word_lower not in {"n't", "'s", "'re", "’t"}:
                        negative_words.append(word_lower)
    return negative_words

# Get positive sentiment words(threshold determines how negative the text is we accept)
def get_strong_positive_words(reviews, threshold=0.3):
    positive_words = []
    for review in reviews:
        blob = TextBlob(review)
        if blob.sentiment.polarity > threshold:
            for word, pos in blob.tags:
                if pos in ['JJ', 'VB', 'RB']:
                    word_lower = word.lower()
                    if word_lower not in {"n't", "'s", "'re", "’t"}:
                        positive_words.append(word_lower)
    return positive_words

# Generate and show word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

file_path = 'dataset_out.json'
df = load_data(file_path)

if 'text' not in df.columns or 'label' not in df.columns:
    print("Error: Dataset must contain 'text' and 'label' columns.")
else:
    positive_reviews = df[df['label'] == 1]['text'].dropna()
    negative_reviews = df[df['label'] == -1]['text'].dropna()

    positive_words = get_strong_positive_words(positive_reviews, threshold=0.3)
    negative_words = get_strong_negative_words(negative_reviews, threshold=-0.3)

    generate_wordcloud(positive_words, "Positive Label Word Cloud")
    generate_wordcloud(negative_words, "Negative Label Word Cloud")
