# Spam Classifier using Logistic Regression and TF-IDF Vectorization🤖

Description:

This Python code implements a basic spam classifier using logistic regression and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique.

The process begins with loading email data from a CSV file, cleaning it, and preparing it for machine learning. The 'Category' column, indicating whether an email is spam or ham, is converted into numerical labels for classification.

Next, the dataset is split into training and testing sets to train the model and evaluate its performance. The TF-IDF vectorizer is employed to convert the text data into numerical features, capturing the importance of each word in the emails.

A logistic regression model is trained on the training data, and its accuracy is assessed on both the training and testing sets.

Finally, the trained model is used to predict whether a sample email (provided as input) is spam or not. The prediction is based on the model's classification, and the result is printed along with an explanation.

Overall, this code provides a fundamental framework for building a spam classifier using machine learning techniques, suitable for simple email filtering tasks.


Code Explanation
~This Python code is for building a simple spam classifier using logistic regression. Let's break down what each part of the code does:

~Imports: This section imports necessary libraries such as NumPy for numerical computing, pandas for data manipulation, and scikit-learn for machine learning functionalities.

~Data Loading: The code reads data from a CSV file named 'mail_data.csv' using pandas read_csv() function and stores it in a DataFrame called df.

~Data Cleaning: Missing values in the DataFrame are filled with empty strings.

~Data Preparation: The 'Category' column values are converted to numerical values. 'spam' is replaced with 0 and 'ham' (which likely means legitimate or non-spam emails) with 1. The 'Message' column is assigned to X and 'Category' to Y.

~Train-Test Split: The dataset is split into training and testing sets using train_test_split() function from scikit-learn. 80% of the data is used for training (X_train and Y_train), and 20% is used for testing (X_test and Y_test).

~Feature Extraction: The TfidfVectorizer from scikit-learn is used to convert text data into numerical features. It converts a collection of raw documents (emails in this case) into a matrix of TF-IDF features.

~Model Training: A logistic regression model is initialized and trained using the training data (X_train_features and Y_train).

~Model Evaluation: The accuracy of the trained model is evaluated on both training and testing datasets using the accuracy_score() function.

~Prediction: Finally, the trained model is used to make predictions on new data. In this case, there's a sample email provided as input_your_mail. The email is converted into features using the same TF-IDF vectorizer, and the model predicts whether it's spam or ham.
