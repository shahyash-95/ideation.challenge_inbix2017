# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:37:08 2017

@author: YASH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:28:01 2017

@author: YASH
"""


import pandas as p 
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as report
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split 

df1=p.read_csv("C:/Users/YASH/Documents/Training-set.csv")
df3=df1.set_index(df1.columns[0])
Y= p.Series(df3["CLASS"])
train_set=df3.drop("CLASS",axis=1)
headers=train_set.columns
#pre processing - Standard Scaling data
X = p.DataFrame(StandardScaler().fit_transform(train_set,),columns=headers)

X1=X.as_matrix()
Y=Y.as_matrix()
##do feature selection 

clf = MLPClassifier(solver="lbfgs", activation="relu", alpha=1, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=True,
                        epsilon=1e-08, hidden_layer_sizes=(5,2), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=500, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)
     
selector = SelectPercentile(f_classif, percentile=30)
X_new=selector.fit_transform(X1,Y)
feature_list=selector.get_support(indices=False)
col=X.columns.tolist()
col_np=np.array(col)
features_selected=col_np[feature_list]
print("features selected",features_selected)
#file = open('C:/Users/YASH/Documents/features.txt','w')  
#file.write(features_selected)  #copy features to text file
#np.savetxt('C:/Users/YASH/Documents/lr-features.txt',features_selected, fmt='%s')
np.savetxt('C:/Users/YASH/Documents/mlp-features.txt',features_selected, fmt='%s', delimiter='\n')

#transform
X_new=X_new.transpose()
Y=Y.transpose()
#Crossvalidation and roc curve
cv =StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
scores = cross_val_score(clf, X_new, Y, scoring='accuracy', cv=cv)
print ("Cross validation scores",scores)
print (scores.mean())


predicted = cross_val_predict(clf, X_new, Y, cv=cv)

fig, ax = plt.subplots()
ax.scatter(Y, predicted, edgecolors=(0, 0, 0))
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('True Values')
ax.set_ylabel('Predictions')
plt.title('Cross validation Evaluation')
plt.show()

### train test split
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.25)

clf.fit(X_train,y_train)

# Cross validation
# X_new--- training data
# x_new_test--- testing
#Testing
y_pred_test=clf.predict(X_test)

acc=report.accuracy_score(y_test,y_pred_test)
print("Test score %.2f" %acc)
print("Classification report Testing")
print (report.classification_report(y_test, y_pred_test)) 

print("Confusion Matrix..") 
print(report.confusion_matrix(y_test, y_pred_test))

pred=clf.predict_proba(X_test)
#accuracy=accuracy_score(y_train,y_pred_train)
#print("Training Accuracy %.2f "%accuracy)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i],thresholds  = roc_curve(y_test, pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

AUC=roc_auc_score(y_test, pred[:,1])
print("Area Under Curve %.2f",AUC)
plt.figure()
plt.plot(fpr[1], tpr[1],label="data 1, auc="+str(AUC))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic- MLP Classifier')
plt.show()
