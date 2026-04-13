import re
import pandas as pd
from nltk.stem import WordNetLemmatizer


# -----------------------------
# Emoji dictionary
# -----------------------------
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire',
    ':(': 'sad', ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry',
    ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked',
    ':-$': 'confused', ':\\': 'annoyed', ':#': 'mute',
    ':X': 'mute', ':^)': 'smile', ':-&': 'confused',
    '$_$': 'greedy', '@@': 'eyeroll', ':-!': 'confused',
    ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile',
    ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel',
    'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}


# -----------------------------
# Stopwords
# -----------------------------
stopwordlist = set([
    'a','about','above','after','again','ain','all','am','an','and','any','are','as','at',
    'be','because','been','before','being','below','between','both','by','can','d','did','do',
    'does','doing','down','during','each','few','for','from','further','had','has','have',
    'having','he','her','here','hers','herself','him','himself','his','how','i','if','in',
    'into','is','it','its','itself','just','ll','m','ma','me','more','most','my','myself',
    'now','o','of','on','once','only','or','other','our','ours','ourselves','out','own',
    're','s','same','she',"shes",'should',"shouldve",'so','some','such','t','than','that',
    "thatll",'the','their','theirs','them','themselves','then','there','these','they','this',
    'those','through','to','too','under','until','up','ve','very','was','we','were','what',
    'when','where','which','while','who','whom','why','will','with','won','y','you',"youd",
    "youll","youre","youve",'your','yours','yourself','yourselves'
])


# -----------------------------
# Main preprocessing function
# -----------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for Sentiment140 dataset.
    """

    df = df.copy()

    # 1. Keep only necessary columns
    df = df[['sentiment', 'text']]

    # 2. Convert labels: 4 → 1 (positive)
    df['sentiment'] = df['sentiment'].replace(4, 1)

    # 3. Clean text
    df['text'] = preprocess_text(df['text'])

    return df


# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess_text(text_data):
    processed_text = []

    word_lemmatizer = WordNetLemmatizer()

    # Regex patterns
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    user_pattern = r"@[^\s]+"
    alpha_pattern = r"[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"

    for tweet in text_data:

        tweet = str(tweet).lower()

        # Replace URLs
        tweet = re.sub(url_pattern, " URL", tweet)

        # Replace emojis
        for emoji in emojis:
            tweet = tweet.replace(emoji, " EMOJI" + emojis[emoji])

        # Replace user mentions
        tweet = re.sub(user_pattern, " USER", tweet)

        # Remove non-alphanumeric
        tweet = re.sub(alpha_pattern, " ", tweet)

        # Reduce repeated characters
        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        tweet_words = []

        for word in tweet.split():

            # Skip stopwords + short words
            if word not in stopwordlist and len(word) > 1:
                word = word_lemmatizer.lemmatize(word)
                tweet_words.append(word)

        processed_text.append(" ".join(tweet_words))

    return processed_text