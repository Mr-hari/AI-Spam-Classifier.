import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 1. Simple Dataset (You can replace this with a CSV later)
data = {
    'text': ['Win a free iPhone now', 'Hey, are we meeting for lunch?', 'Claim your lottery prize', 'Please review the attached document', 'Cheap loans available'],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}
df = pd.DataFrame(data)

# 2. Convert text to numbers (The AI only understands numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# 3. Train the Model
model = MultinomialNB()
model.fit(X, df['label'])

# 4. Test it!
test_msg = ["You won a cash prize!"]
test_vector = vectorizer.transform(test_msg)
prediction = model.predict(test_vector)

print(f"Message: {test_msg[0]}")
print(f"Result: {prediction[0]}")