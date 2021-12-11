import csv
import pandas as pd

from sklearn.model_selection import train_test_split
# for calculation of the confusion matrix
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix
# importing the logistic regression classifier for training on the dataset
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
## datasettwo
data = pd.read_csv('heartdataset.csv')
X = data.drop('stroke', axis=1)

# setting one column as the target label
y = data['stroke']

z = data['stroke']
print("processing...")

# splitting the data into 80 - 20 ratio for getting the labelled training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# getting the solver lbfgs =
clf = LogisticRegression(max_iter=1000,solver='lbfgs')
#training the classifier on the data
clf.fit(X_train, y_train)
# getting predicted value from the classifier
y_pred_test = clf.predict(X_test)
# printing the confitiion matrix
print(confusion_matrix(y_test,y_pred_test))

# calculating the confusion matrix from the precdicted dataset
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_test)

print(confusion_matrix(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# labelling the different axis and the graph
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# lbfgs — Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno.
# It approximates the second derivative matrix updates with gradient evaluations.
# It stores only the last few updates, so it saves memory.
# It isn't super fast with large data sets.