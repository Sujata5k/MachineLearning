









































































import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

try:
    print("Dataset: \n")
    dataset = pd.read_csv(r"F:\suja\Tennis.csv")  # Use raw string
except FileNotFoundError:
    print("Error: File 'Tennis.csv' not found. Please check the path.")
    exit()

x = dataset[['outlook', 'temp', 'humidity', 'windy']]
y = dataset['play']
x = pd.get_dummies(x)

xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.2, random_state=20)

nbc = GaussianNB()
nbc.fit(xtr, ytr)

y_pred = nbc.predict(xt)

print("\n\n\t\t\tClassification Report")
print(classification_report(yt, y_pred))
