import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SentimentModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=2000)  # Reducing to 2000 features
        self.model = SVC()
        self.lemmatizer = WordNetLemmatizer()

        # Load and preprocess the data
        file_path = 'dataset/mental_health.csv'  # Replace with your file path
        data = pd.read_csv(file_path)
        data['basic_processed_text'] = data['text'].apply(self.basic_preprocess_text)

        # Vectorize the text data and get the labels
        X = self.tfidf.fit_transform(data['basic_processed_text'])
        y = data['label'].values

        # Ensure labels are binary and numerical
        y = pd.factorize(y)[0]

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Print the accuracy
        self.print_accuracy()

    def basic_preprocess_text(self, text):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(text))

        # Remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        # Lemmatization
        processed_feature = processed_feature.split()
        processed_feature = [self.lemmatizer.lemmatize(word) for word in processed_feature]
        return ' '.join(processed_feature)

    def predict(self, message):
        # Vectorize the message
        message_vectorized = self.tfidf.transform([message])

        # Predict the sentiment of the message
        result = self.model.predict(message_vectorized)

        # Convert the prediction to a string
        if result[0] == 1:
            result = 'Depression or Anxiety'
        else:
            result = 'No Depression or Anxiety'

        # Print the message and the result
        print(f'Message: {message}')
        print(f'Result: {result}')

        return result, message

    def print_accuracy(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f'Accuracy: {accuracy * 100}%')