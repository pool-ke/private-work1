#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:15:14 2017

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
from sklearn import preprocessing

def onehotencode(arr):
    onehotMat=np.zeros((len(arr),10))
    for i in range(len(arr)):
        j=arr[i]
        onehotMat[int(i),int(j)]=1
    return onehotMat
    




import tensorflow as tf
from numpy.random import RandomState
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size=100
n_hidden1=70
n_hidden2=70
n_hidden3=70
#w1=tf.Variable(tf.random_normal([50,10],stddev=1,seed=1))
#b1=tf.Variable(tf.zeros([10]))
w1=tf.Variable(tf.random_normal([50,n_hidden1],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([n_hidden1,10],stddev=1,seed=1))
b1=tf.Variable(tf.zeros([n_hidden1]))
b2=tf.Variable(tf.zeros([10]))


#w1=tf.Variable(tf.random_normal([50,n_hidden1],stddev=1,seed=1))
#w2=tf.Variable(tf.random_normal([n_hidden1,n_hidden2],stddev=1,seed=1))
#w3=tf.Variable(tf.random_normal([n_hidden2,n_hidden3],stddev=1,seed=1))
#w4=tf.Variable(tf.random_normal([n_hidden3,10],stddev=1,seed=1))
#b1=tf.Variable(tf.zeros([n_hidden1]))
#b2=tf.Variable(tf.zeros([n_hidden2]))
#b3=tf.Variable(tf.zeros([n_hidden3]))
#b4=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,50],name='x-input')
y_=tf.placeholder(tf.float32,[None,10],name='y-input')

hidden1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
y=tf.nn.softmax(tf.matmul(hidden1,w2)+b2)

#hidden1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
#hidden2=tf.nn.sigmoid(tf.matmul(hidden1,w2)+b2)
#hidden3=tf.nn.sigmoid(tf.matmul(hidden2,w3)+b3)
#y=tf.nn.softmax(tf.matmul(hidden3,w4)+b4)
#y=tf.nn.softmax(tf.matmul(x,w1)+b1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

file_path1='mnist_features5000.txt'
file_path2='mnist_labels5000.txt'
data=np.loadtxt(file_path1)
label=np.loadtxt(file_path2)
x_train,x_test,y_train,y_test=train_test_split(data,label,random_state=4,train_size=0.6)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

dataset_size=3000
init_op=tf.initialize_all_variables()
with tf.Session() as sess:
    X=x_train
    X_1=x_test
    Y=onehotencode(y_train)
    Y_1=onehotencode(y_test)
    sess.run(init_op)
    STEPS=50000
    for i in range(STEPS):
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#        print (sess.run(y))
#        print (sess.run(y_))
        if i%200==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print ("After%dtraining steps.cross entropy on all data is %g"%(i,total_cross_entropy))

#            print(accuracy.eval({x:X[start:end],y_:Y[start:end]}))
            print(accuracy.eval({x:X,y_:Y}))
            print(accuracy.eval({x:X_1,y_:Y_1}))
    total_cross_entropy1=sess.run(cross_entropy,feed_dict={x:X_1,y_:Y_1})
    print ("the accuracy of the model on the test data is %g"%(accuracy.eval({x:X_1,y_:Y_1})))
    print ("The cross entropy on test data is %g"%(total_cross_entropy1))

#print (type(X))

#batch_size=8
##
#batch_size=8
#w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
#w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#
#x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
#y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')
#
#a=tf.matmul(x,w1)
#y=tf.matmul(a,w2)
#
#
#cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
#train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#
#rdm=RandomState(1)
#dataset_size=128
#X=rdm.rand(dataset_size,2)
#Y=[[int(x1+x2<1)] for (x1,x2) in X]
#init_op=tf.initialize_all_variables()
#with tf.Session() as sess:
#    sess.run(init_op)
#    STEPS=5000
#    for i in range(STEPS):
#        start=(i*batch_size)%dataset_size
#        end=min(start+batch_size,dataset_size)
#    
#        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#    
#        if i%1000==0:
#            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
#            print ("After%dtraining steps.cross entropy on all data is %g"%(i,total_cross_entropy))
