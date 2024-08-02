import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Import the Dataset
df = pd.read_csv("Phishing_Email.csv")
print(df.head())

# Check NAN values
print(df.isna().sum())

# Drop the Na values
df = df.dropna()
print(df.isna().sum())

# Dataset shape
print(df.shape)

# Count the occurrences of each Email type
email_type_counts = df['Email Type'].value_counts()
print(email_type_counts)

# Create the bar chart
unique_email_types = email_type_counts.index.tolist()
color_map = {'Phishing Email': 'red', 'Safe Email': 'green'}
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]

plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Undersampling technique 
Safe_Email = df[df["Email Type"] == "Safe Email"]
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])

# Check the shape again 
print(Safe_Email.shape, Phishing_Email.shape)

# Create a new DataFrame with balanced Email types
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)
print(Data.head())

# Split the data into features X and Dependent Variable y
X = Data["Email Text"].values
y = Data["Email Type"].values

# Splitting Data 
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", x_test.shape, y_test.shape)

# Define the Random Forest Classifier
rf_classifier = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=10))])

# Train the model
rf_classifier.fit(X_train, y_train)

# Prediction
rf_y_pred = rf_classifier.predict(x_test)

# Print accuracy score, confusion matrix, and classification report
print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred))

# Define the Naive Bayes Classifier
nb_classifier = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])

# Train the Naive Bayes model
nb_classifier.fit(X_train, y_train)

# Prediction for Naive Bayes model
nb_y_pred = nb_classifier.predict(x_test)

# Print Naive Bayes model accuracy
print("Naive Bayes Model Accuracy:", accuracy_score(y_test, nb_y_pred))
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, nb_y_pred))
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_y_pred))