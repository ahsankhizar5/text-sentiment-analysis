import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Step 1: Load dataset
df = pd.read_csv('IMDB_Dataset.csv')  # Ensure the CSV is in the same folder
df.dropna(subset=['review', 'sentiment'], inplace=True)
print("✅ Dataset loaded and cleaned successfully!")
print(df.head())

# Show class balance
print("\n🎯 Sentiment Distribution:\n", df['sentiment'].value_counts())

# Step 2: Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print("\n⏳ Preprocessing text...")
df['clean_review'] = df['review'].apply(preprocess_text)
print("✅ Preprocessing done!")

# Step 3: Convert text to features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])

# Step 4: Encode sentiment (positive → 1, negative → 0)
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation Report:")
print(classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("💾 Model and vectorizer saved!")

# Step 8: Predict custom review
def predict_sentiment(text):
    text = preprocess_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞"

if __name__ == "__main__":
    print("\n📝 Try your own review:")
    sample = input("Enter a movie review: ")
    print("Predicted Sentiment:", predict_sentiment(sample))
