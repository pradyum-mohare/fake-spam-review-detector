Fake Review Detection System

This project is an NLP-based machine learning application that detects whether an e-commerce product review is fake or genuine using TF-IDF and a Linear SVM classifier.
A Streamlit web app is used to provide real-time prediction from user input.

Features

1. Text preprocessing using NLTK

2. TF-IDF vectorization (unigram + bigram)

3. Linear SVM classification (~90% accuracy)

4. Real-time prediction with Streamlit UI

5. Separate training and deployment pipeline

screenshot of UI



Project Structure

train.py        → trains the model and saves model.pkl & tfidf.pkl  
app.py          → Streamlit web application  
reviews.csv     → dataset  
model.pkl       → trained SVM model  
tfidf.pkl       → trained TF-IDF vectorizer  



How to Run
1.Train the model
2.python train.py
3.Run the web app python -m streamlit run app.py
4.Then open http://localhost:8501 in your browser.



Tech Stack

1.Python
2.Scikit-learn
3.NLTK
4.Streamlit
5.pandas


Author

Pradyum Mohare