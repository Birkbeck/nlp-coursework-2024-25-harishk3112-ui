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


def enhanced_political_tokenizer(text):
    """
    Enhanced tokenizer using spaCy for political speech classification.
    
    """
    
    # Political terms and phrases to preserve
    political_terms = {
        'labour', 'conservative', 'liberal', 'democrat', 'tory', 'whig',
        'parliament', 'government', 'minister', 'secretary', 'prime', 
        'chancellor', 'brexit', 'eu', 'uk', 'nhs', 'gdp', 'budget',
        'tax', 'economy', 'policy', 'bill', 'legislation', 'vote',
        'election', 'constituency', 'mp', 'lord', 'commons', 'lords',
        'britain', 'british', 'england', 'scotland', 'wales', 'ireland'
    }
    
    # Important political phrases, convert to single tokens
    political_phrases = {
        'prime minister': 'primeminister',
        'foreign secretary': 'foreignsecretary',
        'home secretary': 'homesecretary',
        'shadow minister': 'shadowminister',
        'member of parliament': 'memberofparliament',
        'house of commons': 'houseofcommons',
        'house of lords': 'houseoflords',
        'civil service': 'civilservice',
        'public sector': 'publicsector',
        'private sector': 'privatesector',
        'social security': 'socialsecurity',
        'health service': 'healthservice',
        'national health service': 'nationalhealthservice'
    }
    
    # Important sentiment and modal words for political context
    important_modifiers = {
        'not', 'never', 'always', 'must', 'should', 'will', 'would', 'could',
        'support', 'oppose', 'against', 'favour', 'believe', 'think',
        'agree', 'disagree', 'increase', 'decrease', 'reform', 'change',
        'improve', 'reduce', 'strengthen', 'weaken', 'more', 'less',
        'better', 'worse', 'higher', 'lower', 'great', 'important'
    }
    
    # Preprocess text
    text_processed = text.lower()
    
    # Replace political phrases with single tokens
    for phrase, replacement in political_phrases.items():
        text_processed = text_processed.replace(phrase, replacement)
    
    # Expand contractions
    text_processed = contractions.fix(text_processed)
    
    # Process with spaCy
    doc = nlp(text_processed)
    
    processed_tokens = []
    
    for token in doc:
        # Skip whitespace, punctuation, and very short tokens
        if token.is_space or token.is_punct or len(token.text) < 2:
            continue
            
        # Get lemmatized form
        lemma = token.lemma_.lower().strip()
        
        # Skip empty lemmas
        if not lemma or len(lemma) < 2:
            continue
            
        # Handle different cases
        if token.like_url or token.like_email:
            continue
        elif token.is_digit:
            # Keep years as special tokens
            if len(token.text) == 4 and token.text.startswith(('19', '20')):
                processed_tokens.append('YEAR')
            continue
        elif lemma in political_terms:
            # Always keep political terms
            processed_tokens.append(lemma)
        elif lemma in important_modifiers:
            # Keep important modifiers and sentiment words
            processed_tokens.append(lemma)
        elif (token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV'] and 
              not token.is_stop and 
              len(lemma) > 2 and
              any(c.isalpha() for c in lemma)):
            # Keep meaningful content words
            processed_tokens.append(lemma)
        elif token.pos_ == 'PROPN' and len(lemma) > 2:
            # Keep proper nouns 
            processed_tokens.append(lemma)
    
    return processed_tokens

# Create custom vectorizer with optimized parameters
vectorizer_custom = TfidfVectorizer(
    max_features=3000,
    tokenizer=enhanced_political_tokenizer,
    lowercase=False,
    token_pattern=None,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm='l2'
)

# Get the train and test text data using the same indices from your existing splits
# We need to reconstruct the original text data for the same train/test split
train_indices = X_train_ngram.nonzero()[0]  # Get indices from existing split
test_indices = X_test_ngram.nonzero()[0]

# Create arrays to get the same train/test split as before
df_reset = df.reset_index(drop=True)
X_speech = df_reset["speech"]
y_speech = df_reset["party"]

# Use the same train_test_split as before to get the same indices
X_train_text, X_test_text, y_train_custom, y_test_custom = train_test_split(
    X_speech, y_speech, test_size=0.2, stratify=y_speech, random_state=26
)

# Now apply the custom vectorizer
X_train_custom = vectorizer_custom.fit_transform(X_train_text)
X_test_custom = vectorizer_custom.transform(X_test_text)

# Use your already trained classifiers , parameters but retrain with custom features
rf_custom = RandomForestClassifier(n_estimators=300, random_state=42)
svm_custom = SVC(kernel="linear", random_state=42)

# Train with custom features
rf_custom.fit(X_train_custom, y_train_custom)
svm_custom.fit(X_train_custom, y_train_custom)

# Make predictions
rf_custom_preds = rf_custom.predict(X_test_custom)
svm_custom_preds = svm_custom.predict(X_test_custom)

# Calculate F1 scores
rf_f1 = f1_score(y_test_custom, rf_custom_preds, average='macro')
svm_f1 = f1_score(y_test_custom, svm_custom_preds, average='macro')

print("\nQ2(e) Custom Tokenizer Results:")
print(f"Random Forest F1 Score: {round(rf_f1, 4)}")
print(f"SVM F1 Score: {round(svm_f1, 4)}")

# Print classification report for the best performing classifier
if rf_f1 > svm_f1:
    print(f"\nBest Performing Classifier: Random Forest")
    print("Classification Report:")
    print(classification_report(y_test_custom, rf_custom_preds))
else:
    print(f"\nBest Performing Classifier: SVM")
    print("Classification Report:")
    print(classification_report(y_test_custom, svm_custom_preds))