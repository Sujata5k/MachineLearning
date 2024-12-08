






































































































import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Set working directory
os.chdir(r"F:\suja")

# Import dataset
data_csv = pd.read_csv('ass1.csv', na_values=["?"])
print(data_csv)
print("----------------------------------------------------------------")
data_csv.info()
print("----------------------------------------------------------------")
data_csv.isna().sum()
print(data_csv.isna().sum())
print("----------------------------------------------------------------")

# Print missing data information
print(data_csv.isnull().sum())
print("----------------------------------------------------------------")

# Find rows with missing data
missing = data_csv[data_csv.isnull().any(axis=1)]
print(missing)
print("----------------------------------------------------------------")

# Show dataset description
data_csv.describe()
print(data_csv.describe())
print("----------------------------------------------------------------")

# Handling missing data (fill missing values with mean, median, and mode)
data_csv['Age'] = data_csv['Age'].fillna(data_csv["Age"].mean())
data_csv['Income'] = data_csv['Income'].fillna(data_csv['Income'].median())
data_csv['Region'] = data_csv['Region'].fillna(data_csv['Region'].mode()[0])

# Verify no missing values after filling
print(data_csv.isnull().sum())
print("----------------------------------------------------------------")

# Encode categorical data
df = pd.DataFrame(data_csv)

# Fill missing data using mean for numerical columns
df.fillna({'Age': df['Age'].mean(), 'Income': df['Income'].mean()}, inplace=True)

# Label encoding for categorical variable
label_encoder = LabelEncoder()
df['Online Shopper'] = label_encoder.fit_transform(df['Online Shopper'])

# One hot encoding for 'Region' column
one_hot_encoder = OneHotEncoder()
region_encoded = one_hot_encoder.fit_transform(df[['Region']]).toarray()
region_encoded_df = pd.DataFrame(region_encoded, columns=one_hot_encoder.get_feature_names_out(['Region']))

# Concatenate one-hot encoded columns and drop original 'Region' column
df = pd.concat([df, region_encoded_df], axis=1).drop(['Region'], axis=1)

print("Encoded DataFrame:")
print(df)
print()

# Perform train-test split
X = df.drop("Online Shopper", axis=1)
y = df["Online Shopper"]

# Train-test split with 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print("Train-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Feature scaling (Standardizing features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display first 5 rows of scaled training data
print("Feature Scaled Data (First 5 rows of X_train_scaled):")
print(X_train_scaled[:5])
