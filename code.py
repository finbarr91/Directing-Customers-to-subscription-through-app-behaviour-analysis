# Directing-Customers-to-subscription-through-app-behaviour-analysis
Directing Customers to subscription through app behaviour analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
dataset = pd.read_csv('new_appdata10.csv')
print(dataset.head())


# DATA PREPROCESSING
response = dataset['enrolled']
dataset = dataset.drop(columns= 'enrolled')

X_train, X_test,y_train, y_test = train_test_split(dataset,response,test_size=0.2,random_state=0)

train_identifier = X_train['user']
X_train = X_train.drop(columns = 'user')
test_identifier = X_test['user']
X_test = X_test.drop(columns = 'user')

sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values

X_train = X_train2
X_test = X_test2

# Model Building
classifier = LogisticRegression (random_state=0, penalty='l2')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, fmt='g',annot=True)
plt.show()

df_cm = pd.DataFrame(cm,index=(0,1), columns = (0,1))
plt.figure(figsize= (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm,annot=True, fmt='g')
plt.show()
print("\nTest Data Accuracy : %0.4f\n" % accuracy_score(y_test,y_pred))

print('\naccuracy score:\n',accuracy_score(y_test, y_pred))
print('\n Precision score:\n', precision_score(y_test,y_pred))
print('\nRecall score:\n', recall_score(y_test, y_pred))
print('\n F1 score:\n', f1_score(y_test,y_pred))
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Logistic Accuracy: %0.3f (+/- %0.3f)'% (accuracies.mean(), accuracies.std()*2))

# FORMATTING THE FINAL RESULT
final_results = pd.concat([y_test,test_identifier], axis =1).dropna()
final_results['predicted_result'] = y_pred
final_results[['user', 'enrolled', 'predicted_result']].reset_index(drop=True)
print(final_results)



