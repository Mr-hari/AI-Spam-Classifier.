# 1. Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 2. Dataset
data = {
    'text': [
        'Win a free iPhone now',
        'Hey, are we meeting for lunch?',
        'Claim your lottery prize',
        'Please review the attached document',
        'Cheap loans available',
        'Congratulations! You won a lottery',
        'Let us catch up tomorrow',
        'Free entry in 2 lakh prize',
        'Project meeting at 10 AM',
        'Get cash bonus now'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'ham', 'spam', 'ham', 'spam'
    ]
}

df = pd.DataFrame(data)

# 3. Convert Text to Numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Check Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Test with New Message
test_msg = ["You won a cash prize! Click now"]
test_vector = vectorizer.transform(test_msg)
prediction = model.predict(test_vector)

print(f"\nMessage: {test_msg[0]}")
print(f"Prediction: {prediction[0]}")
