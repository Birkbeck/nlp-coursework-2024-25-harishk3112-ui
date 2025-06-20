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

def nltk_ttr(df):
    """
    Tokenizes each novel and calculates the type-token ratio.
    Adds a new column 'ttr' to the DataFrame and returns it.
    """
    ttr_values = []

    for text in df["text"]:
        tokens = text.lower().split()
        if len(tokens) == 0:
            ttr = 0
        else:
            ttr = len(set(tokens)) / len(tokens)
        ttr_values.append(ttr)

    df["ttr"] = ttr_values
    return df

    
def flesch_kincaid(df):
    """
    Returns a dictionary mapping the title of each novel to the
    Flesch-Kincaid reading grade level score of the text.
    Uses the NLTK library for tokenization and the CMU pronouncing
    dictionary for estimating syllable counts.
    """
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import cmudict

    cmu = cmudict.dict()
    scores = {}

    for _, row in df.iterrows():
        title = row["title"]
        text  = row["text"]

        # 1. Sentence and word tokenization
        sentences = sent_tokenize(text)
        words     = [w for w in word_tokenize(text) if w.isalpha()]

        num_sent  = len(sentences)
        num_words = len(words)

        # 2. Syllable counting via CMU dict
        total_syllables = 0
        for w in words:
            wl = w.lower()
            if wl in cmu:
                # take first pronunciation variant
                pron = cmu[wl][0]
                total_syllables += sum(1 for ph in pron if ph[-1].isdigit())
            else:
                total_syllables += 1

        # 3. Compute Flesch-Kincaid Grade Level
        if num_sent == 0 or num_words == 0:
            fk = 0.0
        else:
            fk = (
                0.39 * (num_words / num_sent)
              + 11.8 * (total_syllables / num_words)
              - 15.59
            )

        scores[title] = round(fk, 2)

    return scores









if __name__ == "__main__":
    df = read_novels()
    df = nltk_ttr(df)
    fk_scores = flesch_kincaid(df)
    print(fk_scores)


