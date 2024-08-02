import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import os

# Create a directory to save the plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

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
plt.savefig(os.path.join(output_dir, 'email_type_distribution.png'))
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
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_class_report = classification_report(y_test, rf_y_pred)

print("Random Forest Model Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:\n", rf_conf_matrix)
print("Random Forest Classification Report:\n", rf_class_report)

# Define the Naive Bayes Classifier
nb_classifier = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])

# Train the Naive Bayes model
nb_classifier.fit(X_train, y_train)

# Prediction for Naive Bayes model
nb_y_pred = nb_classifier.predict(x_test)

# Print Naive Bayes model accuracy
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_conf_matrix = confusion_matrix(y_test, nb_y_pred)
nb_class_report = classification_report(y_test, nb_y_pred)

print("Naive Bayes Model Accuracy:", nb_accuracy)
print("Naive Bayes Confusion Matrix:\n", nb_conf_matrix)
print("Naive Bayes Classification Report:\n", nb_class_report)

# Plot accuracy comparison
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Random Forest', 'Naive Bayes']
accuracies = [rf_accuracy, nb_accuracy]
ax.bar(labels, accuracies, color=['blue', 'orange'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison.png'))
plt.show()

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].matshow(rf_conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(rf_conf_matrix.shape[0]):
    for j in range(rf_conf_matrix.shape[1]):
        axes[0].text(x=j, y=i, s=rf_conf_matrix[i, j], va='center', ha='center')

axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xlabel('Predicted labels')
axes[0].set_ylabel('True labels')

axes[1].matshow(nb_conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(nb_conf_matrix.shape[0]):
    for j in range(nb_conf_matrix.shape[1]):
        axes[1].text(x=j, y=i, s=nb_conf_matrix[i, j], va='center', ha='center')

axes[1].set_title('Naive Bayes Confusion Matrix')
axes[1].set_xlabel('Predicted labels')
axes[1].set_ylabel('True labels')

plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
plt.show()

# Plot classification reports
rf_report = classification_report(y_test, rf_y_pred, output_dict=True)
nb_report = classification_report(y_test, nb_y_pred, output_dict=True)

fig, ax = plt.subplots(figsize=(12, 8))
labels = list(rf_report.keys())[:-3]
rf_precision = [rf_report[label]['precision'] for label in labels]
nb_precision = [nb_report[label]['precision'] for label in labels]
rf_recall = [rf_report[label]['recall'] for label in labels]
nb_recall = [nb_report[label]['recall'] for label in labels]
rf_f1 = [rf_report[label]['f1-score'] for label in labels]
nb_f1 = [nb_report[label]['f1-score'] for label in labels]

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, rf_precision, width, label='RF Precision')
rects2 = ax.bar(x + width/2, nb_precision, width, label='NB Precision')

ax.set_xlabel('Labels')
ax.set_title('Precision Comparison by Label')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_comparison.png'))
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))

rects1 = ax.bar(x - width/2, rf_recall, width, label='RF Recall')
rects2 = ax.bar(x + width/2, nb_recall, width, label='NB Recall')

ax.set_xlabel('Labels')
ax.set_title('Recall Comparison by Label')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'recall_comparison.png'))
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))

rects1 = ax.bar(x - width/2, rf_f1, width, label='RF F1-Score')
rects2 = ax.bar(x + width/2, nb_f1, width, label='NB F1-Score')

ax.set_xlabel('Labels')
ax.set_title('F1-Score Comparison by Label')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'f1score_comparison.png'))
plt.show()
