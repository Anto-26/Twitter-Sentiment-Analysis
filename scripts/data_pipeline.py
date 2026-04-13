from src.dataloading.load_data import load_data
from src.preprocessing.text_cleaning import preprocess_dataframe
from src.features.tfidf_vectoriser import get_tfidf_vectoriser
from src.models.train import train_model
import joblib

def main():
    dataset = load_data("/Users/jamesantoarnoldj/Desktop/Projects/TSA/data/raw/Dataset.csv")
    clean_df = preprocess_dataframe(dataset)
    vectorizer = get_tfidf_vectoriser()
    X = vectorizer.fit_transform(clean_df["text"])
    y = clean_df["sentiment"]


    model = train_model(X, y)

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nPipeline executed successfully!")
    print("Model + Vectorizer saved.")

if __name__ == "__main__":
    main()