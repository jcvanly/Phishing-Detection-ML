import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Import the Dataset
df = pd.read_csv("Phishing_Email.csv")

# Drop NaN values
df = df.dropna()

# Undersampling technique
Safe_Email = df[df["Email Type"] == "Safe Email"]
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0], random_state=42)

# Create a new DataFrame with balanced Email types
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)

# Split the data into features X and Dependent Variable y
X = Data["Email Text"].values
y = Data["Email Type"].values

# Splitting Data
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Random Forest Classifier with TfidfVectorizer
rf_classifier = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])

# Train the model
rf_classifier.fit(X_train, y_train)

# Extract the feature importances
rf_model = rf_classifier.named_steps['classifier']
tfidf = rf_classifier.named_steps['tfidf']
feature_importances = rf_model.feature_importances_
feature_names = tfidf.get_feature_names_out()

# Get predictions for the training set to understand the contributions of each word
X_train_transformed = tfidf.transform(X_train)
phishing_indices = np.where(y_train == 'Phishing Email')[0]
safe_indices = np.where(y_train == 'Safe Email')[0]

phishing_importance = np.mean(X_train_transformed[phishing_indices].toarray(), axis=0)
safe_importance = np.mean(X_train_transformed[safe_indices].toarray(), axis=0)

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances,
    'Phishing': phishing_importance,
    'Safe': safe_importance
})

importance_df['Class'] = np.where(importance_df['Phishing'] > importance_df['Safe'], 'Phishing', 'Safe')
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)  # Get top 20 important words

# Plot the most important words
plt.figure(figsize=(10, 8))
colors = ['red' if cls == 'Phishing' else 'green' for cls in importance_df['Class']]
plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
plt.xlabel('Importance')
plt.ylabel('Words')
plt.title('Top 20 Important Words in Determining Phishing or Safe Email')
plt.gca().invert_yaxis()
plt.show()

