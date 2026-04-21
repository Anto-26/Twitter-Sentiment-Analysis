# Twitter Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active-green)
![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning project that classifies tweet sentiment (positive/negative) using NLP techniques trained on the Sentiment140 dataset. Includes a Streamlit web app for live predictions.

---

## Features

- Preprocesses raw tweets: removes URLs, mentions, emoji conversion, lemmatization, and stopword filtering
- Extracts features using TF-IDF vectorization
- Trains a Logistic Regression classifier (~80–85% accuracy)
- Saves the trained model and vectorizer as `.pkl` files
- Interactive Streamlit app for real-time sentiment prediction

---

## Project Structure

```
TSA/
├── app/
│   └── app.py                  # Streamlit web app
├── data/
│   └── raw/                    # Raw dataset (not tracked)
├── models/
│   ├── sentiment_model.pkl     # Trained Logistic Regression model
│   └── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   └── preprocessing.ipynb     # Preprocessing experiments
├── scripts/
│   └── data_pipeline.py        # End-to-end training pipeline
├── src/
│   ├── dataloading/
│   │   └── load_data.py
│   ├── preprocessing/
│   │   └── text_cleaning.py    # Tweet cleaning & lemmatization
│   ├── features/
│   │   └── tfidf_vectoriser.py
│   └── models/
│       └── train.py            # Model training & evaluation
├── requirements.txt
└── README.md
```

---

## Dataset

**Sentiment140** — 1.6 million tweets collected via the Twitter API.

| Column | Description |
|--------|-------------|
| `sentiment` | 0 = Negative, 4 = Positive |
| `ids` | Tweet ID |
| `date` | Timestamp |
| `flag` | Query flag |
| `user` | Twitter username |
| `text` | Tweet content |

Labels are remapped during preprocessing: `4 → 1` (Positive), `0` stays Negative.

Download the dataset from [Kaggle — Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it at `data/raw/Dataset.csv`.

---

## ML Pipeline

```
Raw tweets
  └── Text cleaning
        ├── Lowercase
        ├── URL removal
        ├── User mention removal (@user → USER)
        ├── Emoji → text mapping
        ├── Non-alphanumeric removal
        ├── Repeated character reduction
        ├── Stopword filtering
        └── WordNet lemmatization
  └── TF-IDF vectorization
  └── Logistic Regression (max_iter=1000)
  └── Evaluation: accuracy + classification report
```

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd TSA

# Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate        # macOS/Linux
myenv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (run once)
python -c "import nltk; nltk.download('wordnet')"
```

---

## Usage

### Train the model

```bash
python scripts/data_pipeline.py
```

This will:
1. Load the dataset from `data/raw/Dataset.csv`
2. Clean and preprocess all tweets
3. Fit the TF-IDF vectorizer and transform the text
4. Train a Logistic Regression model and print accuracy + classification report
5. Save `models/sentiment_model.pkl` and `models/vectorizer.pkl`

### Run the Streamlit app

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, enter any tweet, and click **Predict Sentiment**.

---

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80–85% |

Exact results vary with preprocessing settings and train/test split seed.

---

## Requirements

See [requirements.txt](requirements.txt) for the full list. Key dependencies:

- `scikit-learn` — model training and TF-IDF
- `nltk` — WordNet lemmatizer
- `pandas` / `numpy` — data handling
- `streamlit` — web app
- `joblib` — model serialization
- `matplotlib` / `seaborn` / `wordcloud` — EDA visualizations

---

## Author

**James Anto Arnold James Sagayaraj**

---

## License

This project is licensed under the [MIT License](LICENSE).
