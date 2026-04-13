from src.data.load_data import load_data
from src.preprocessing.text_cleaning import preprocess_dataframe

dataset = load_data("/Users/jamesantoarnoldj/Desktop/Projects/TSA/data/raw/Dataset.csv")
clean_df = preprocess_dataframe(dataset)

print(clean_df.head())
