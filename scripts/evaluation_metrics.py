from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
# import spacy

# Load spaCy model
# nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Function to preprocess text if needed
    return text.lower()

def calculate_tfidf_cosine_similarity(text1, text2):
    # Preprocess the text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Calculate TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

def calculate_jaccard_similarity(text1, text2):
    set1 = set(preprocess_text(text1).split())
    set2 = set(preprocess_text(text2).split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_score = float(len(intersection)) / len(union)
    return jaccard_score

def get_sentiment(text):
    blob = TextBlob(preprocess_text(text))
    return blob.sentiment.polarity

def calculate_tone_similarity(text1, text2):
    sentiment1 = get_sentiment(text1)
    sentiment2 = get_sentiment(text2)
    tone_similarity = 1 - abs(sentiment1 - sentiment2)
    return tone_similarity
"""
def get_entities(text):
    doc = nlp(preprocess_text(text))
    return set([ent.text for ent in doc.ents])

def calculate_accuracy_similarity(text1, text2):
    entities1 = get_entities(text1)
    entities2 = get_entities(text2)
    if not entities1 and not entities2:
        return 1.0  # If no entities are found, assume they are equally accurate
    accuracy_similarity = len(entities1.intersection(entities2)) / len(entities1.union(entities2))
    return accuracy_similarity
"""
def calculate_conciseness_ratio(text1, text2):
    length_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    return length_ratio

def evaluate_strings(text1, text2):
    evaluation_metrics = {
        "content_similarity_tfidf_cosine": calculate_tfidf_cosine_similarity(text1, text2),
        "content_similarity_jaccard": calculate_jaccard_similarity(text1, text2),
        "tone_similarity": calculate_tone_similarity(text1, text2),
        # "accuracy_similarity": calculate_accuracy_similarity(text1, text2),
        "conciseness_ratio": calculate_conciseness_ratio(text1, text2),
    }
    return evaluation_metrics
