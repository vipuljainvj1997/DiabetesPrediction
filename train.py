import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=names)

print(df.dtypes)

# split X and Y
X = df.iloc[:, :8]
y = df.iloc[:, 8]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=101)

# train the model
model = LogisticRegression()
model.fit(X_train, y_train)
print("[INFO] Model Trainig Completed")

#evaluate the model
result = model.score(X_test, y_test)
print(f"[INFO] Model score : {result}")

joblib.dump(model, "dib_model.pkl")