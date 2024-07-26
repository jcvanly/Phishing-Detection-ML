# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
dataset_path = 'Phishing_Email.csv'
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isna().sum())

# Drop rows with missing values
df = df.dropna()

# Display dataset shape
print("Dataset shape after dropping missing values:", df.shape)

# Count the occurrences of each email type
email_type_counts = df['Email Type'].value_counts()
print(email_type_counts)

# Create a bar chart for email type distribution
unique_email_types = email_type_counts.index.tolist()
color_map = {'Phishing Email': 'red', 'Safe Email': 'green'}
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]

plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Handle class imbalance using undersampling
Safe_Email = df[df["Email Type"] == "Safe Email"]
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)

# Split the data into features (X) and target variable (y)
X = Data["Email Text"].values
y = Data["Email Type"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the RandomForestClassifier model with a pipeline
classifier = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=100, random_state=0))])

# Train the model
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(classifier, 'phishing_email_classifier.pkl')

# Function to predict fraud from a list of emails
def predict_fraud(email_texts):
    model = joblib.load('phishing_email_classifier.pkl')
    predictions = model.predict(email_texts)
    return predictions

# Example usage
emails_to_check = ["Your account has been suspended. Click here to verify your details.", 
                   "Meeting tomorrow at 10 AM. Please confirm your availability.",
                   "Dear Customer, We noticed suspicious activity on your account. Please click the link below to verify your information immediately." ,
                   "Hi Team, Attached is the latest update on the project. Please review and provide your feedback by end of the day." ,
                   "Hey, Are we still on for dinner tonight at 7 PM? Let me know if you need to reschedule.",
                   "enron enron enron",
                   "your, your, your"
                   ]

predictions = predict_fraud(emails_to_check)
for email, pred in zip(emails_to_check, predictions):
    print(f"Email: {email}\nPrediction: {pred}\n")

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cross_val_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", np.mean(cross_val_scores))

# Predict probabilities for ROC and Precision-Recall Curves
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label="Phishing Email")
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob, pos_label="Phishing Email")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Safe Email", "Phishing Email"], yticklabels=["Safe Email", "Phishing Email"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature Importance
if hasattr(classifier.named_steps['classifier'], 'feature_importances_'):
    importances = classifier.named_steps['classifier'].feature_importances_
    feature_names = classifier.named_steps['tfidf'].get_feature_names_out()
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette=sns.color_palette('viridis', len(feature_importance_df)))
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.show()


# Learning Curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(classifier, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score', color='darkorange')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', color='navy')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.tight_layout()
plt.show()
