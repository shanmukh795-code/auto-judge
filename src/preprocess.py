import re

def combine_text(df):
    """
    Combine title, description, input and output text into one column
    """
    df["full_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["input_description"].fillna("") + " " +
        df["output_description"].fillna("")
    )
    return df


def clean_text(text):
    """
    Clean a single text string
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df):
    """
    Full preprocessing pipeline
    """
    df = combine_text(df)
    df["full_text"] = df["full_text"].apply(clean_text)
    return df

