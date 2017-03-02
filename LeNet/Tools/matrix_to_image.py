#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding=gbk
from PIL import Image
import numpy as np
import pandas as pd
# pip install lmdb
import lmdb
import os 
import caffe
import matplotlib.pyplot as plt
from  sklearn.cross_validation import train_test_split

#visuliza a digit
#get the local file localtion
path= os.getcwd() +'/'

train_path=path+'train.csv'
test_path=path+'test.csv'
train_df=pd.read_csv(train_path)#42000
test_df=pd.read_csv(test_path)#28000
train_np=train_df.values
y_train=train_np [:,0]
print train_df.values
X_train=train_np [:,1:]
print "X_train:",X_train
X_train=X_train.reshape((X_train.shape[0],1,28,28))
# X_test=X_test.reshape((X_test.shape[0],1,28,28))

im1=X_train[9,0]


def MatrixToImage(matrix):
    #matrix = matrix*255
    new_im = Image.fromarray(matrix.astype(np.uint8))
    #new_im = Image.fromarray(matrix)
    return new_im

if(os.path.exists(path+'kaggle_train_label.txt')):
    os.remove(path+'kaggle_train_label.txt')
f=open(path+'kaggle_train_label.txt','a')

for i in range(20):
	im1=X_train[i,0]
	print im1.shape
	new_im = MatrixToImage(im1)
	print 'new_im',new_im
	plt.imshow(im1, cmap='gray')
	new_im.save(path+'mnist_kaggle_train_1/'+str(i)+'.bmp',cmap='gray')

	f.write(str(i)+'.bmp '+str(y_train[i])+'\n')
