










































































import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

# File path for the dataset
file_path = r'F:\suja\Emails.csv'

# Check if the file exists
if os.path.exists(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Explore the dataset
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Handle missing values using forward fill
    df.ffill(inplace=True)

    # Check for 'label' column
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the DataFrame.")
        print("Available columns are:", df.columns.tolist())
    else:
        # Encode the label
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])

        # Ensure 'label' column is the target and the rest are features
        X = df.drop('label', axis=1)
        y = df['label']

        # Encode categorical features if necessary
        X = pd.get_dummies(X, drop_first=True)

        # Check if all features are numeric after encoding
        if not np.issubdtype(X.dtypes, np.number).all():
            print("Error: All features must be numeric.")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Best estimator
        knn = grid_search.best_estimator_

        # Make predictions
        y_pred = knn.predict(X_test)

        # Calculate accuracy and precision
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')

        print(f'\nAccuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))

        # Classify a new email
        new_email_features = [[10, 1, 0]]  # Replace with actual feature values
        new_email_features = scaler.transform(new_email_features)
        new_email_prediction = knn.predict(new_email_features)

        print('\nNew Email Classification:')
        print('Spam' if le.inverse_transform(new_email_prediction)[0] == 'spam' else 'Not Spam')
else:
    print(f"Error: The file at {file_path} was not found.")
