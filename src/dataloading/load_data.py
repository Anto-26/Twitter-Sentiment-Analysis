import pandas as pd

DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from the csv file.
    """

    try:
        df = pd.read_csv(
            file_path,
            encoding = DATASET_ENCODING,
            names = DATASET_COLUMNS
        )
        if df.empty:
            raise ValueError("Dataset is empty")
        
        print(f"Dataset is loaded Successfully")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    