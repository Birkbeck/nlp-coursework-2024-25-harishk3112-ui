import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression


import re
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import contractions

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Q2(a) : Data Preprocessing

# Load the dataset
df = pd.read_csv("PartTwo/hansard40000.csv")

# Rename 'Labour (Co-op)' to 'Labour'
df["party"] = df["party"].replace("Labour (Co-op)", "Labour")

# Keep only top 4 most frequent parties (excluding 'Speaker')
top_parties = df["party"].value_counts().drop("Speaker", errors="ignore").head(4)
df = df[df["party"].isin(top_parties.index)]

# Keep only rows where 'speech_class' == 'Speech'
df = df[df["speech_class"] == "Speech"]

# Keep only rows where 'speech' text is at least 1000 characters
df = df[df["speech"].str.len() >= 1000]

# Print the final shape
print("Filtered DataFrame shape:", df.shape)

# Q2(b) : Train/Test  Data Split

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Apply vectorizer to the 'speech' column
X = vectorizer.fit_transform(df["speech"])

# Extract labels
y = df["party"]

# Split data using stratified sampling with seed 26
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=26
)

# Print result sizes
print(" TF-IDF shape:", X.shape)
print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# Q2(c) : Model Training 

# Train a Random Forest classifier with 300 trees
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)  
rf_predictions = rf.predict(X_test) 

# Print results for Random Forest
print("\nQ2(c)  Random Forest Results:")
print("Macro F1 Score:", round(f1_score(y_test, rf_predictions, average='macro'), 4))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Train a SVM classifier with linear kernel
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train, y_train)  
svm_predictions = svm.predict(X_test)  

# Print results for SVM
print("\nQ2(c)  SVM Results:")
print("Macro F1 Score:", round(f1_score(y_test, svm_predictions, average='macro'), 4))
print("Classification Report:\n", classification_report(y_test, svm_predictions))

# Q2(d) : Train classifiers using TF-IDF with unigrams, bigrams, and trigrams (max 3000 features)

# Create new TF-IDF vectorizer with unigrams, bigrams, and trigrams
vectorizer_ngram = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3))

# Fit and transform the speech data
X_ngram = vectorizer_ngram.fit_transform(df["speech"])

# Train/test split again with ngram features
X_train_ngram, X_test_ngram, y_train_ngram, y_test_ngram = train_test_split(
    X_ngram, y, test_size=0.2, stratify=y, random_state=26
)

# Train Random Forest
rf_ngram = RandomForestClassifier(n_estimators=300, random_state=42)
rf_ngram.fit(X_train_ngram, y_train_ngram)
rf_ngram_preds = rf_ngram.predict(X_test_ngram)

# Print results
print("\nQ2(d)  Random Forest with ngram (1,3) Results:")
print("Macro F1 Score:", round(f1_score(y_test_ngram, rf_ngram_preds, average='macro'), 4))
print("Classification Report:\n", classification_report(y_test_ngram, rf_ngram_preds))

# Train SVM
svm_ngram = SVC(kernel="linear", random_state=42)
svm_ngram.fit(X_train_ngram, y_train_ngram)
svm_ngram_preds = svm_ngram.predict(X_test_ngram)

# Print results
print("\nQ2(d)  SVM with ngram (1,3) Results:")
print("Macro F1 Score:", round(f1_score(y_test_ngram, svm_ngram_preds, average='macro'), 4))
print("Classification Report:\n", classification_report(y_test_ngram, svm_ngram_preds))


# Q2(e) : Evaluate models with custom tokenizer

# Custom tokenizer with spaCy, contractions, and named entities
def spacy_tokenizer(text):
    expanded_text = contractions.fix(text)
    doc = nlp(expanded_text)
    # Lemmatized tokens
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
        and token.is_alpha
    ]
    # Named entities
    entities = [ent.text.lower() for ent in doc.ents]
    return tokens + entities

#  Custom tokenizer, n-grams, and feature extraction
vectorizer = TfidfVectorizer(
    tokenizer=spacy_tokenizer,
    max_features=3000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    stop_words=None  # Handled in tokenizer
)

X = vectorizer.fit_transform(df["speech"])
y = df["party"]

# Train/test split (use same random_state as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=26
)

# Train and evaluate both models (SVM and Random Forest)
# Trained 2 models as per the question it explicitely mentioned for the above used classifiers.
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel="linear", random_state=42, class_weight='balanced')
}

best_score = 0
best_clf_name = ""
best_preds = None

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)  # Use original (not oversampled) data
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    if f1 > best_score:
        best_score = f1
        best_clf_name = name
        best_preds = y_pred

# Print only the best model results
print("\nBest Classifier:", best_clf_name)
print("Macro F1 Score:", round(best_score, 4))
print("Classification Report:\n", classification_report(y_test, best_preds))




