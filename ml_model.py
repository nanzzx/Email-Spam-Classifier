import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data: fill missing values, convert labels to numerical values.
    """
    data = df.where((pd.notnull(df)), '')
    data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})
    return data

def train_model(data):
    """
    Train the logistic regression model on the given dataset.
    """
    X = data['Message']
    Y = data['Category']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    
    return model, feature_extraction, X_test, Y_test

def evaluate_model(model, feature_extraction, X_test, Y_test):
    """
    Evaluate the trained model on the test data and return accuracy.
    """
    X_test_features = feature_extraction.transform(X_test)
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
    return accuracy_on_test_data

def predict_spam(model, feature_extraction, email_content):
    """
    Predict whether the given email content is spam or ham.
    """
    input_data_features = feature_extraction.transform([email_content])
    prediction = model.predict(input_data_features)
    return prediction[0]

if __name__ == "__main__":
    # Example usage:
    # Load data
    df = load_data('mail_data.csv')
    
    # Preprocess data
    data = preprocess_data(df)
    
    # Train model
    model, feature_extraction, X_test, Y_test = train_model(data)
    
    # Evaluate model
    accuracy = evaluate_model(model, feature_extraction, X_test, Y_test)
    print("Accuracy on test data:", accuracy)
