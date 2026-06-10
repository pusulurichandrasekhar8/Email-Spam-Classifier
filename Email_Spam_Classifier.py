from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Congratulations! You won a prize",
    "Claim your free gift now",
    "Meeting at 10 AM tomorrow",
    "Project submission deadline"
]

labels = ["spam", "spam", "ham", "ham"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

test_email = ["Free gift waiting for you"]

X_test = vectorizer.transform(test_email)
prediction = model.predict(X_test)

print("Prediction:", prediction[0])