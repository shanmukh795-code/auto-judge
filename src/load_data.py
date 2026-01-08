import json
import pandas as pd

def load_data(path):
    """
    Load JSONL dataset and return as pandas DataFrame
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))

    return pd.DataFrame(data)

