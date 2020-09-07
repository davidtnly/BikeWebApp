import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
import pickle


data = pd.read_csv('https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv')

data['date'] = data['date'].apply(pd.to_datetime)
data['year'] = [i.year for i in data['date']]
data['month'] = [i.month_name()[0:3] for i in data['date']]
data['day'] = [i.day_name()[0:3] for i in data['date']]

x = data[['temperature', 'humidity', 'windspeed']]
y = data['count']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=20)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')
rp = sns.regplot(x=y_test, y=predictions)

with open('model/bike_model_xgboost.pkl', 'wb') as file:
    pickle.dump(classifier, file)