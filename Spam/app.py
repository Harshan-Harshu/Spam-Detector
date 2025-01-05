from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv("SMSSpamCollection.txt", sep='\t', names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        # Get user input
        new_sms = request.form['sms']
        # Transform the input using the vectorizer
        new_sms_vector = vectorizer.transform([new_sms])
        # Make a prediction
        prediction = nb.predict(new_sms_vector)[0]
        result = "This SMS is Spam" if prediction == 1 else "This SMS is not spam"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
