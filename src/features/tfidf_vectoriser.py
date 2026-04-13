from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectoriser():
    """
    Create and return a TF-IDF vectoriser
    """

    vectoriser = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2)
    )

    return vectoriser