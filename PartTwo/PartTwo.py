import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report



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
print("\nQ2(c) – Random Forest Results:")
print("Macro F1 Score:", round(f1_score(y_test, rf_predictions, average='macro'), 4))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Train a SVM classifier with linear kernel
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train, y_train)  
svm_predictions = svm.predict(X_test)  

# Print results for SVM
print("\nQ2(c) – SVM Results:")
print("Macro F1 Score:", round(f1_score(y_test, svm_predictions, average='macro'), 4))
print("Classification Report:\n", classification_report(y_test, svm_predictions))