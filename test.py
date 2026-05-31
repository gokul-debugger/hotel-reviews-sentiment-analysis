# ============================================================
# Hotel Reviews Sentiment Analysis
# test.py — Live review sentiment predictor
# Run: streamlit run test.py
# ============================================================

import streamlit as st
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================================
# Load Models
# ============================================================

@st.cache_resource
def load_models():
    vectorizer_pos = joblib.load('Models/tfidf_vectorizer_positive.pkl')
    vectorizer_neg = joblib.load('Models/tfidf_vectorizer_negative.pkl')
    model_pos = joblib.load('Models/logistic_regression_positive.pkl')
    model_neg = joblib.load('Models/logistic_regression_negative.pkl')
    return vectorizer_pos, vectorizer_neg, model_pos, model_neg

vectorizer_pos, vectorizer_neg, model_pos, model_neg = load_models()

# ============================================================
# Text Cleaning
# ============================================================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    stemmer = nltk.stem.PorterStemmer()
    text = ' '.join([stemmer.stem(w) for w in text.split()])
    return text

# ============================================================
# Sentiment emoji helper
# ============================================================

def sentiment_emoji(sentiment):
    return {'positive': '😊 Positive', 'neutral': '😐 Neutral',
            'negative': '😞 Negative'}.get(sentiment, sentiment)

def sentiment_color(sentiment):
    return {'positive': '#2ecc71', 'neutral': '#3498db',
            'negative': '#e74c3c'}.get(sentiment, '#999')

# ============================================================
# App Layout
# ============================================================

st.set_page_config(page_title='Hotel Review Predictor',
                   page_icon='🏨', layout='centered')

st.title('🏨 Hotel Review Sentiment Predictor')
st.markdown('**La Veranda Hotel — Booking.com Reviews**')
st.markdown('---')

st.markdown('### Enter your review below')
st.markdown('Provide what you liked and what you didn\'t like separately — just like on Booking.com!')

col1, col2 = st.columns(2)

with col1:
    positive_input = st.text_area(
        '👍 What did you like?',
        placeholder='e.g. The room was spotless and staff were very friendly...',
        height=180
    )

with col2:
    negative_input = st.text_area(
        '👎 What did you dislike?',
        placeholder='e.g. The pool was closed and breakfast was disappointing...',
        height=180
    )

st.markdown('---')

if st.button('🔍 Analyse Sentiment', use_container_width=True):
    if not positive_input.strip() and not negative_input.strip():
        st.warning('Please enter at least one review before analysing.')
    else:
        st.markdown('## Results')
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown('### 👍 Positive Review Analysis')
            if positive_input.strip():
                cleaned = clean_text(positive_input)
                vectorized = vectorizer_pos.transform([cleaned])
                prediction = model_pos.predict(vectorized)[0]
                color = sentiment_color(prediction)
                st.markdown(
                    f'<div style="background-color:{color}22; border-left: 5px solid {color}; '
                    f'padding:15px; border-radius:5px;">'
                    f'<h3 style="color:{color}; margin:0">{sentiment_emoji(prediction)}</h3>'
                    f'<p style="margin:5px 0 0 0; color:#555">"{positive_input[:100]}..."</p>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.info('No positive review entered.')

        with col_neg:
            st.markdown('### 👎 Negative Review Analysis')
            if negative_input.strip():
                cleaned = clean_text(negative_input)
                vectorized = vectorizer_neg.transform([cleaned])
                prediction = model_neg.predict(vectorized)[0]
                color = sentiment_color(prediction)
                st.markdown(
                    f'<div style="background-color:{color}22; border-left: 5px solid {color}; '
                    f'padding:15px; border-radius:5px;">'
                    f'<h3 style="color:{color}; margin:0">{sentiment_emoji(prediction)}</h3>'
                    f'<p style="margin:5px 0 0 0; color:#555">"{negative_input[:100]}..."</p>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.info('No negative review entered.')

st.markdown('---')
st.markdown('*Powered by Logistic Regression trained on La Veranda Hotel reviews*')