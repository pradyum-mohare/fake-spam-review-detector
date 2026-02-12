import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

st.set_page_config(page_title="Fake Review Detector",page_icon="🛍️",layout="centered"
)

st.title("🛍️ Fake Review Detection System")
st.markdown(
    "This NLP model analyzes product reviews and predicts whether the review is **Fake** or **Genuine**."
)
st.divider()

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6FAkJZA_ooQDxeenPe7FiTNfk8K2kX6vgbg&s', width= 300)
def predict_review(text):
    clean = preprocess(text)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]
    if prediction == 1:
        return "Fake Review"
    else:
        return "Real Review"



review = st.text_area("Enter Product Review",placeholder="Type or paste a product review here...",height=150)
if review:
    st.write("checking")
    
result = predict_review(review)
st.write("Prediction:", result)