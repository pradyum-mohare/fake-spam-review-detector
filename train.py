import pandas as pd
import numpy as np
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
review = pd.read_csv('reviews.csv')

review = review.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
review.label = review.label.apply(lambda x: 1 if x == 'cg' else 0 if x=='or' else x)
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    text = text.lower()                      
    text = re.sub(r'[^a-z\s]', '', text)     
    tokens = word_tokenize(text)             
    tokens = [w for w in tokens if w not in stop_words]  
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   
    return " ".join(tokens)
review["clean_text"] = review["text_"].apply(remove_stopwords)
from sklearn.feature_extraction.text import TfidfVectorizer

# create TF-IDF object
tfidf =TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),      # unigrams + bigrams
    min_df=2,               # ignore rare words
    max_df=0.9              # ignore too common words
)


# convert clean text into numbers
X = tfidf.fit_transform(review["clean_text"])

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, review["label"], test_size=0.2, random_state=42)
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

y_pred =svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import pickle

pickle.dump(svm_model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))







