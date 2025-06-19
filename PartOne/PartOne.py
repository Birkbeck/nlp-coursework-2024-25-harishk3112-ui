#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.
import os
import pandas as pd

import nltk
import spacy
from pathlib import Path


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def read_novels(path=Path("PartOne/novels")):
    """
    Reads texts from a directory of .txt files and returns a DataFrame
    with the text, title, author, and year.
    """
    rows = []

    for file in path.glob("*.txt"):
        try:
            title, author, year = file.stem.split("-")
            year = int(year)
        except ValueError:
            print(f"Skipping file with unexpected name format: {file.name}")
            continue

        with open(file, encoding="utf-8") as f:
            text = f.read()

        rows.append({
            "text": text,
            "title": title,
            "author": author,
            "year": year
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="year").reset_index(drop=True)
    return df



if __name__ == "__main__":
    df = read_novels()
    print(df.head())
