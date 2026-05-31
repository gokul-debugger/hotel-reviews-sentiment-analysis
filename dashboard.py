# ============================================================
# Hotel Reviews Sentiment Analysis
# dashboard.py — Main Streamlit Dashboard
# Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import charts

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title='Hotel Sentiment Dashboard',
    page_icon='🏨',
    layout='wide'
)

# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv('Dataset/cleaned/hotel_reviews_cleaned.csv')
    df['VisitDate'] = pd.to_datetime(df['VisitDate'], errors='coerce')
    return df.dropna(subset=['sentiment'])

df = load_data()

# ============================================================
# Header
# ============================================================

st.title('🏨 Hotel Reviews Sentiment Analysis Dashboard')
st.markdown('**La Veranda Hotel — Booking.com | Larnaca, Cyprus**')
st.markdown('---')

# ============================================================
# KPI Cards
# ============================================================

total = len(df)
positive_pct = round(df[df['sentiment'] == 'positive']['sentiment'].count() / total * 100, 1)
negative_pct = round(df[df['sentiment'] == 'negative']['sentiment'].count() / total * 100, 1)
neutral_pct  = round(df[df['sentiment'] == 'neutral']['sentiment'].count() / total * 100, 1)
avg_score    = round(df['Score'].mean(), 2)
countries    = df['GuestCountry'].nunique()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('📝 Total Reviews', f'{total:,}')
col2.metric('😊 Positive', f'{positive_pct}%')
col3.metric('😞 Negative', f'{negative_pct}%')
col4.metric('⭐ Avg Score', f'{avg_score}/10')
col5.metric('🌍 Countries', countries)

st.markdown('---')

# ============================================================
# Sidebar Filters
# ============================================================

st.sidebar.title('🔍 Filters')

sentiment_filter = st.sidebar.multiselect(
    'Sentiment',
    options=['positive', 'neutral', 'negative'],
    default=['positive', 'neutral', 'negative']
)

group_filter = st.sidebar.multiselect(
    'Group Type',
    options=df['GroupType'].dropna().unique().tolist(),
    default=df['GroupType'].dropna().unique().tolist()
)

score_filter = st.sidebar.slider(
    'Score Range',
    min_value=1, max_value=10,
    value=(1, 10)
)

# Apply filters
filtered_df = df[
    (df['sentiment'].isin(sentiment_filter)) &
    (df['GroupType'].isin(group_filter)) &
    (df['Score'].between(score_filter[0], score_filter[1]))
]

st.sidebar.markdown(f'**Showing {len(filtered_df):,} of {total:,} reviews**')

# ============================================================
# Row 1 — Sentiment Overview
# ============================================================

st.subheader('📊 Sentiment Overview')
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('**Sentiment Distribution**')
    st.plotly_chart(charts.sentiment_donut(filtered_df),
                    use_container_width=True, key='donut')

with col2:
    st.markdown('**Score Distribution**')
    st.plotly_chart(charts.score_distribution(filtered_df),
                    use_container_width=True, key='score')

with col3:
    st.markdown('**Polarity Distribution**')
    st.plotly_chart(charts.polarity_distribution(filtered_df),
                    use_container_width=True, key='polarity')

st.markdown('---')

# ============================================================
# Row 2 — Guest Insights
# ============================================================

st.subheader('🌍 Guest Insights')
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Top 10 Guest Countries**')
    st.plotly_chart(charts.top_countries(filtered_df),
                    use_container_width=True, key='countries')

with col2:
    st.markdown('**Sentiment by Group Type**')
    st.plotly_chart(charts.sentiment_by_group(filtered_df),
                    use_container_width=True, key='group')

st.markdown('---')

# ============================================================
# Row 3 — Room & Time Analysis
# ============================================================

st.subheader('🛏️ Room & Time Analysis')
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Average Score by Room Type**')
    st.plotly_chart(charts.sentiment_by_room(filtered_df),
                    use_container_width=True, key='room')

with col2:
    st.markdown('**Reviews Over Time by Sentiment**')
    st.plotly_chart(charts.reviews_over_time(filtered_df),
                    use_container_width=True, key='time')

st.markdown('---')

# ============================================================
# Row 4 — Word Frequency
# ============================================================

st.subheader('💬 Most Common Words')
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Positive Reviews — Top Words**')
    st.plotly_chart(charts.wordcloud_treemap(filtered_df, 'cleaned_positive'),
                    use_container_width=True, key='pos_words')

with col2:
    st.markdown('**Negative Reviews — Top Words**')
    st.plotly_chart(charts.wordcloud_treemap(filtered_df, 'cleaned_negative'),
                    use_container_width=True, key='neg_words')

st.markdown('---')

# ============================================================
# Row 5 — Raw Data Table
# ============================================================

st.subheader('📋 Review Data')
show_cols = ['PositiveReview', 'NegativeReview', 'Score',
             'sentiment', 'GuestCountry', 'GroupType', 'RoomType', 'VisitDate']
st.dataframe(filtered_df[show_cols].head(50), use_container_width=True)

st.markdown('---')
st.markdown('*Dashboard built with Streamlit · Dataset: La Veranda Hotel, Booking.com*')