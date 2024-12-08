









































































import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Feature2': [2, 3, 1, 4, 5, 3, 2, 4, 5],
    'Target': [0, 1, 0, 1, 1, 0, 0, 1, 1],
}

df = pd.DataFrame(data)

# Separate features and target variable
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split the dataset into training and testing sets with a smaller test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)

# Display the results
print("Random Forest Classifier Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
