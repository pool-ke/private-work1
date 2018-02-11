#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:59:54 2017

@author: root
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt



if __name__ == "__main__":
    file_path1='mnist_features5000.txt'
    file_path2='mnist_labels5000.txt'
    x=np.loadtxt(file_path1)
    y=np.loadtxt(file_path2)
    print (x.shape)
    print (y.shape)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,train_size=0.6)
    

    '''************AdaBoost Classifier*************'''
#    bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),algorithm="SAMME",n_estimators=100,
#                           learning_rate=0.8)
#    bdt.fit(x_train,y_train)
#
#    y_pre1=bdt.predict(x_train)
#    y_pre2=bdt.predict(x_test)
#    print (y_train)
#    print (y_pre1)
#    print (bdt.score(x_train,y_train))
#    print (y_test)
#    print (y_pre2)
#    print (bdt.score(x_test,y_test))
    '''************SVM Classify****************'''
    clf=svm.SVC(C=0.8,kernel='rbf',gamma=0.0023,decision_function_shape='ovr')
#    scores=cross_val_score(clf,x_train,y_train,cv=10)
#    print (scores)
#    clf=svm.SVC(C=10,kernel='linear')
#    Carr=np.linspace(0.5,0.9,5)
#    gammaarr=np.linspace(0.0021,0.0025,5)
#    param_grid=dict(C=Carr,gamma=gammaarr)
#    grid =GridSearchCV(clf,param_grid,cv=10,scoring='accuracy')
#    grid.fit(x_train,y_train)
#    print (grid.grid_scores_)
    
    '''accuracy on training and testing'''
    clf.fit(x_train,y_train)
    y_hat1=clf.predict(x_train)
    y_hat2=clf.predict(x_test)
    print (y_train)
    print (y_hat1)
    print (clf.score(x_train,y_train))
    print (y_test)
    print (y_hat2)
    print (clf.score(x_test,y_test))
    clf.fit(x_train,y_train)
    y_hat2=clf.predict(x_train)    

    '''classification report and confusion matrix'''
#    clf.fit(x_train,y_train)
#    y_hat=clf.predict(x_test)
#    print (y_test)
#    print (y_hat)
#    print (clf.score(x_test,y_test))
#    target_names=['class 0','class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']
#    print (classification_report(y_test,y_hat,target_names=target_names))
#    labels=list(set(y_test))
#    confusion_mat=confusion_matrix(y_test,y_hat,labels=labels)
#    print ("confusion_matrix(left labels:y_true,up labels:y_pred):")
#    print ("labels\t",end='')
#    for i in range(len(labels)):
#        print (labels[i],"\t",end='')
#    print ()
#    for i in range(len(confusion_mat)):
#        print (i,"\t",end=''),
#        for j in range(len(confusion_mat[i])):
#            print (confusion_mat[i][j],'\t',end='')
#        print ()
#    print ()
#    

    
    
    '''************RandomForest Classify****************'''
#    clf=RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=9)
#    clf.fit(x_train,y_train)
#    y_hat1=clf.predict(x_train)
#    y_hat2=clf.predict(x_test)
#    print (y_hat1)
#    print (clf.score(x_train,y_train))
#    print (y_hat2)
#    print (clf.score(x_test,y_test))   


#    model=svm.SVR(kernel='rbf')
#    c_can=np.logspace(-2,2,2)
#    gamma_can=np.logspace(-2,2,2)
#    svr=GridSearchCV(model,param_grid={'C':c_can,'gamma':gamma_can},cv=5)



    
#    print (precision_score(y_test,y_hat2))
#    print (recall_score(y_test,y_hat2))
#    print (f1_score(y_test,y_hat2))
#    print (clf.predict(x_train))
#    print (clf.predict(x_test))
#    print (clf.n_support_)
#    print (clf.dual_coef_)
#    print (clf.support_)
#    print (clf.score(x_test,y_test))
#img_vec1=np.loadtxt(file_path1)
#img_vec2=np.loadtxt(file_path2)
#
#print (int(img_vec2[0]))
#print (img_vec1.shape)
#a=np.where(labels2==9)
#print (len(a[0]))
#np.savetxt(out_path1,sample_features_5000)
#np.savetxt(out_path2,sample_labels_5000)
#img_vec_OK=img_vec[0:100,:]
#img_vec_OK_test=img_vec[100:120,:]
#img_vec_NG_test=np.loadtxt(file_path2)
#n_OK_test=img_vec_OK_test.shape[0]
#n_NG_test=img_vec_NG_test.shape[0]
