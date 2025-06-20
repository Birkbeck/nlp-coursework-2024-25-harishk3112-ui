#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import os
import pandas as pd
import math

import nltk
import spacy
from pathlib import Path
from collections import Counter

from nltk.tokenize import word_tokenize
import string




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
    Returns a dictionary mapping each novel title to its type-token ratio (TTR).
    Tokenization is done using NLTK. Ignores case and punctuation.
    """
    ttr_scores = {}

    for _, row in df.iterrows():
        title = row["title"]
        text = row["text"]

        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        if tokens:
            ttr = len(set(tokens)) / len(tokens)
        else:
            ttr = 0.0

        ttr_scores[title] = round(ttr, 4)

    return ttr_scores

    
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



def parse(df, pickle_path=Path("PartOne/parsed_novels.pkl")):
    """
    Q1(e): Parse each novel’s text with spaCy, pickle the DataFrame,
    then load it back and return it (with a new 'doc' column of Doc objects).
    """
    # 1. run nlp() over each raw text
    parsed_docs = [ nlp(text) for text in df["text"] ]

    # 2. attach to a copy of the DataFrame
    df2 = df.copy()
    df2["doc"] = parsed_docs

    # 3. write out & immediately reload from pickle
    print(f"→ parse(): writing pickle to {pickle_path}")
    df2.to_pickle(pickle_path)
    print("✓ parse(): pickle created")
    return pd.read_pickle(pickle_path)




def subjects_by_verb_count(doc, verb):
    """
    Q1(f)(2): Extract the most common subjects of a given verb in a parsed document.
    Returns a list of (subject, count) tuples.
    """
    subj_counter = Counter()
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.lemma_ == verb:
            subj_counter[token.text.lower()] += 1
    return subj_counter.most_common()


def object_counts(doc):
    """
    Q1(f)(1): Extract the most common syntactic objects in a parsed document.
    Returns a list of (object, count) tuples.
    """
    obj_counter = Counter()
    for token in doc:
        if token.dep_ in ("dobj", "obj"):
            obj_counter[token.text.lower()] += 1
    return obj_counter.most_common()


def subjects_by_verb_pmi(doc, target_verb):
    """
    Q1(f)(3): Computes PMI between subjects and a specific verb.
    Returns list of (subject, PMI) tuples, sorted by PMI descending.
    """
    subj_verb_pairs = Counter()
    subj_counts = Counter()
    verb_counts = Counter()

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.lemma_ == target_verb:
            subj = token.text.lower()
            subj_verb_pairs[subj] += 1
            subj_counts[subj] += 1
            verb_counts[target_verb] += 1

    total = sum(subj_verb_pairs.values())
    pmi_scores = []

    for subj in subj_verb_pairs:
        joint = subj_verb_pairs[subj] / total
        p_subj = subj_counts[subj] / total
        p_verb = verb_counts[target_verb] / total

        if p_subj > 0 and p_verb > 0:
            pmi = math.log2(joint / (p_subj * p_verb))
            pmi_scores.append((subj, round(pmi, 3)))

    return sorted(pmi_scores, key=lambda x: x[1], reverse=True)






if __name__ == "__main__":
    # Q1(a): Read novels
    df = read_novels()
    print("\nQ1(a) – First 5 rows of the novels dataframe:")
    print(df[["title", "author", "year"]].head())

    # Q1(b): TTR using NLTK
    ttr_scores = {}
    for i, row in df.iterrows():
        tokens = nltk.word_tokenize(row["text"].lower())
        tokens = [t for t in tokens if t.isalpha()]
        ttr = len(set(tokens)) / len(tokens) if tokens else 0
        ttr_scores[row["title"]] = round(ttr, 4)
    print("\nQ1(b) – Type-Token Ratios (TTR):")
    for title, score in ttr_scores.items():
        print(f"{title}: {score}")

    # Q1(c): Flesch-Kincaid scores
    fk_scores = flesch_kincaid(df)
    print("\nQ1(c) – Flesch-Kincaid Reading Grade Level:")
    for title, score in fk_scores.items():
        print(f"{title}: {score}")

    # Q1(e): Parse and pickle novels
    df = parse(df)

    # Q1(f)(1): Most common syntactic objects
    print("\nQ1(f)(1) – Top 10 syntactic objects in each novel:")
    for _, row in df.iterrows():
        print(f"\n{row['title']}")
        for obj, count in object_counts(row["doc"])[:10]:
            print(f"{obj}: {count}")

    # Q1(f)(2): Most common subjects of verb 'hear'
    print("\nQ1(f)(2) – Top 10 subjects of verb 'hear':")
    for _, row in df.iterrows():
        print(f"\n{row['title']}")
        for subj, count in subjects_by_verb_count(row["doc"], "hear")[:10]:
            print(f"{subj}: {count}")

    # Q1(f)(3): PMI scores for subjects of 'hear'
    print("\nQ1(f)(3) – PMI scores for subjects of 'hear':")
    for _, row in df.iterrows():
        print(f"\n{row['title']}")
        for subj, pmi in subjects_by_verb_pmi(row["doc"], "hear")[:10]:
            print(f"{subj}: {pmi}")


   


