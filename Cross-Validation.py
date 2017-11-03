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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

no_of_features=[10,25,40,55,66]
features_used=[]
acc_score=[]
prec_score=[]
rcall_score=[]
f_score=[]
head=["Feature Subset","Accuracy Score","Precion Score","Recall","F1 score"]

df1=p.read_csv("C:/Users/YASH/Documents/Training-set.csv")
df2=p.read_csv("C:/Users/YASH/Documents/Testing-set.csv")
df3=df1.set_index(df1.columns[0])
df4=df2.set_index(df2.columns[0])
y_train = p.Series(df3["CLASS"])
y_test =p.Series(df4["CLASS"])
train_set=df3.drop("CLASS",axis=1)
test_set=df4.drop("CLASS",axis=1)
headers=train_set.columns
#pre processing - Normalization of data
x_train = p.DataFrame(Normalizer().fit_transform(train_set,),columns=headers)
x_test = p.DataFrame(Normalizer().fit_transform(test_set),columns=headers)
features_dict = {}
##do feature selection 
for i in no_of_features:
    selector = SelectKBest(f_classif, k=i)
    selector.fit(x_train,y_train)
    X_featured = selector.transform(x_train)
    feature_names = list(x_train.columns[selector.get_support(indices=True)])
    #print(feature_names)
    features_dict[i]=feature_names
    features_used.append(i)             
#neural network mlp classifier 

    clf = MLPClassifier(solver="lbfgs", activation="relu", alpha=1, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=True,
                        epsilon=1e-08, hidden_layer_sizes=(10,5), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=500, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)
                
                
    clf.fit(X_featured,y_train)
    y_pred=clf.predict(X_featured)
    accuracy = accuracy_score(y_train,y_pred)
    acc_score.append(accuracy)
    precision = precision_score(y_train, y_pred, average='micro')
    prec_score.append(precision)
    f1 = f1_score(y_train, y_pred, average='micro')
    f_score.append(f1)
    recall=recall_score(y_train, y_pred, average='micro')
    rcall_score.append(recall)
    print("Training Accuracy score",accuracy)
    #add to dataframe 
    
finalscore_analysis=p.DataFrame(np.column_stack([features_used,acc_score,prec_score,rcall_score,f_score]),columns=head)
finalscore_analysis.to_csv("C:/Users/YASH/Documents/Cross Validation.csv")
 
print("Testing....")   
y_predtest=clf.predict(x_test)
Test_accuracy = accuracy_score(y_test,y_predtest)
print("Test Acccuracy",Test_accuracy)
test_precision = precision_score(y_test, y_predtest, average='micro')
print("Precion Score Testing",test_precision)
test_f1 = f1_score(y_test, y_predtest, average='micro')
print("F1 score",test_f1 )
test_recall=recall_score(y_test, y_predtest, average='micro')
print("Recall Score Testing",test_recall)

