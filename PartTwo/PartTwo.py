import pandas as pd

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
