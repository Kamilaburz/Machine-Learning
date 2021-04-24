import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_file = pd.read_csv('Advertising.csv')

X = train_file[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Female']]
y = train_file['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))


