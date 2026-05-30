# Hotel Reviews Sentiment Analysis

A complete sentiment analysis pipeline on 1,500 customer reviews from La Veranda Hotel on Booking.com.

## Notebooks

| Notebook | Description |
|---|---|
| 1_dataPreprocessing | Data cleaning, text preprocessing, sentiment labeling |
| 2_machineLearning | TF-IDF + 6 ML classifiers on positive & negative review tracks |
| 3_deepLearning | LSTM on both review tracks |
| 4_LLM | DistilBERT zero-shot sentiment analysis + model comparison |

## Dataset

- **Source:** [Kaggle - La Veranda Hotel Booking.com Reviews](https://www.kaggle.com/datasets/michelhatab/hotel-reviews-bookingcom/data)
- **Size:** 1,500 reviews
- **Features:** Positive Review, Negative Review, Score, Guest Country, Room Type, Visit Date

## Key Approach

- Separate analysis of Positive and Negative review tracks
- Score-based sentiment labeling (1-4 negative, 5-7 neutral, 8-10 positive)
- Comparison across ML, Deep Learning and LLM approaches

## Setup

```bash
python3.12 -m venv sentiment_env
source sentiment_env/bin/activate
pip install -r requirements.txt
```
