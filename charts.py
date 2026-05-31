# ============================================================
# Hotel Reviews Sentiment Analysis
# charts.py — Reusable chart functions for the dashboard
# ============================================================

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
import string


def sentiment_donut(df):
    """Donut chart showing sentiment distribution."""
    counts = df['sentiment'].value_counts()
    colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.6,
        marker_colors=[colors.get(s, '#999') for s in counts.index]
    )])
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13)
    )
    return fig


def score_distribution(df):
    """Bar chart showing distribution of guest scores."""
    score_counts = df['Score'].value_counts().sort_index()
    fig = px.bar(
        x=score_counts.index,
        y=score_counts.values,
        color=score_counts.values,
        color_continuous_scale='RdYlGn',
        labels={'x': 'Score', 'y': 'Number of Reviews'}
    )
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13),
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=False, linecolor='black')
    fig.update_yaxes(showgrid=False, linecolor='black')
    return fig


def top_countries(df):
    """Horizontal bar chart of top 10 guest countries."""
    top = df['GuestCountry'].value_counts().head(10)
    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation='h',
        color=top.values,
        color_continuous_scale='Blues',
        labels={'x': 'Number of Reviews', 'y': 'Country'}
    )
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13),
        coloraxis_showscale=False,
        yaxis=dict(autorange='reversed')
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def sentiment_by_group(df):
    """Stacked bar chart of sentiment by group type."""
    grouped = df.groupby(['GroupType', 'sentiment']).size().reset_index(name='count')
    colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    fig = px.bar(
        grouped,
        x='GroupType',
        y='count',
        color='sentiment',
        barmode='stack',
        color_discrete_map=colors,
        labels={'count': 'Number of Reviews', 'GroupType': 'Group Type'}
    )
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def sentiment_by_room(df):
    """Bar chart of average score by room type."""
    room_avg = df.groupby('RoomType')['Score'].mean().sort_values(ascending=False)
    fig = px.bar(
        x=room_avg.index,
        y=room_avg.values,
        color=room_avg.values,
        color_continuous_scale='RdYlGn',
        labels={'x': 'Room Type', 'y': 'Average Score'}
    )
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13),
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=False)
    return fig


def reviews_over_time(df):
    """Line chart of reviews over time by sentiment."""
    df = df.copy()
    df['VisitDate'] = pd.to_datetime(df['VisitDate'], errors='coerce')
    df['year_month'] = df['VisitDate'].dt.to_period('M').astype(str)
    colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    fig = go.Figure()
    for sentiment in ['positive', 'neutral', 'negative']:
        subset = df[df['sentiment'] == sentiment].groupby('year_month').size().reset_index(name='count')
        fig.add_trace(go.Scatter(
            x=subset['year_month'],
            y=subset['count'],
            mode='lines+markers',
            name=sentiment.capitalize(),
            line=dict(color=colors[sentiment])
        ))
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13),
        xaxis_title='Month',
        yaxis_title='Number of Reviews'
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def wordcloud_treemap(df, column='cleaned_positive'):
    """Treemap word frequency chart for a given text column."""
    nltk.download('stopwords', quiet=True)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    text = df[column].astype(str).str.lower()
    text = text.str.replace('[{}]'.format(string.punctuation), ' ', regex=True)
    all_words = ' '.join(text).split()
    all_words = [w for w in all_words if w not in stop_words and len(w) > 2]

    word_counts = pd.Series(all_words).value_counts().head(80)
    word_df = pd.DataFrame({'word': word_counts.index, 'count': word_counts.values})

    fig = px.treemap(
        word_df,
        path=['word'],
        values='count',
        color='count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        paper_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=False
    )
    return fig


def polarity_distribution(df):
    """Side by side polarity histograms for positive and negative tracks."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['polarity_positive'],
        name='Positive Reviews',
        marker_color='#2ecc71',
        opacity=0.75,
        nbinsx=20
    ))
    fig.add_trace(go.Histogram(
        x=df['polarity_negative'],
        name='Negative Reviews',
        marker_color='#e74c3c',
        opacity=0.75,
        nbinsx=20
    ))
    fig.update_layout(
        barmode='overlay',
        paper_bgcolor='#f7f7f7',
        plot_bgcolor='#f7f7f7',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family='Arial', size=13),
        xaxis_title='Polarity Score',
        yaxis_title='Count'
    )
    return fig