import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset loaded successfully!")
print(df.head())  # just to check

# Rename columns (IMDB dataset already has correct names, but safe)
df.columns = ["text", "label"]

# Reduce size for faster training (optional but recommended)
df = df.sample(5000)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

print("Text cleaned!")

# Convert text → numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])

# Train model
model = LogisticRegression()
model.fit(X, df["label"])

print("Model trained!")

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")