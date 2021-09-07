# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:12:44 2021

@author: SHASHANK
"""

import pandas as pd
df = pd.read_csv("breast-cancer-wisconsin-data.csv") 
df

df.shape
list(df)

df.dtypes
df.info()

##############################################################

#label encoder target variable is in Descrete form
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Class'] = LE.fit_transform(df['diagnosis'])

list(df)

pd.crosstab(df.diagnosis, df.Class)

################################################################################

#Selecting X and y

X = df.iloc[:,2:32]
X

Y = df['Class']
Y
#######################################################################

#train_test_split

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = .30, random_state = 12)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

#######################################################################

#implementing the model

from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)

logreg.intercept_
logreg.coef_

y_pred = logreg.predict(X_test)
y_pred

#######################################################################
#import metric class

from sklearn import metrics
cm = metrics.confusion_matrix(Y_test,y_pred)
cm

metrics.accuracy_score(Y_test,y_pred).round(2)
metrics.recall_score(y_pred,Y_test)
#######################################################################
#plot
import matplotlib.pyplot as plt
plt.matshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel('predicted label')
plt.xlabel('True label')
plt.show()

