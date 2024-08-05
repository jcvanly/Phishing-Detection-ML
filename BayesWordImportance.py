import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

# Define the Naive Bayes Classifier with TfidfVectorizer
nb_classifier = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])

# Train the model
nb_classifier.fit(X_train, y_train)

# Extract the feature importances
tfidf = nb_classifier.named_steps['tfidf']
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
    'Phishing': phishing_importance,
    'Safe': safe_importance
})

importance_df['Class'] = np.where(importance_df['Phishing'] > importance_df['Safe'], 'Phishing', 'Safe')
importance_df = importance_df.sort_values(by='Phishing', ascending=False).head(20)  # Get top 20 important words

# Plot the most important words
plt.figure(figsize=(10, 8))
colors = ['red' if cls == 'Phishing' else 'green' for cls in importance_df['Class']]
plt.barh(importance_df['Feature'], importance_df['Phishing'], color=colors)
plt.xlabel('Phishing Importance')
plt.ylabel('Words')
plt.title('Top 20 Important Words in Determining Phishing or Safe Email')
plt.gca().invert_yaxis()
plt.show()

# Function to predict if a new email is phishing or safe
def predict_email_type(email_text):
    prediction = nb_classifier.predict([email_text])
    return prediction[0]

# Example usage of the prediction function
email1 = "The USPS package has arrived at the warehouse and cannot be delivered due to incomplete address information. Please confirm your address and account information in the link within 12 hours. https://uspsfos.info/us (please click the link  or copy the link to a browser to open it) the US postal system wishes you a wonderful day https://uspsfos.info/us"
email2 = "Hi Susan, are you still able to make our 10 AM standup on Tuesday? If not, please reach out to me asap and we can reschedule."
email3 = "You and your empty account to the website.  For the time in this is urgent. We here want our product to be with you. Click on product.com."
result1 = predict_email_type(email1)
result2 = predict_email_type(email2)
result3 = predict_email_type(email3)


print(email1 + "\n")
print(f"The above email is predicted to be: {result1} \n")

print(email2 + "\n")
print(f"The above email is predicted to be: {result2} \n")

print(email3 + "\n")
print(f"The above email is predicted to be: {result3} \n")
