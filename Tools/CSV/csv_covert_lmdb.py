#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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
y=train_np [:,0]
X=train_np [:,1:]
X=X.reshape((X.shape[0],1,28,28))
im1=X[10,0]
print y[10]
plt.rcParams['image.cmap']='gray'
#plt.imshow(im1)
#Show the image in csv
#plt.show()


#converting
def covert_lmdb(X,y,path):
    m=X.shape[0]
    map_size=X.nbytes*10#donot worry , mapsize no harm
    # http://lmdb.readthedocs.io/en/release/#environment-class
    env=lmdb.open(path,map_size=map_size)
    # http://lmdb.readthedocs.io/en/release/#lmdb.Transaction
    with env.begin(write=True) as txn:
        for i in range(m):
            datum=caffe.proto.caffe_pb2.Datum()
            datum.channels=X.shape[1]
            datum.height=X.shape[2]
            datum.width=X.shape[3]
            datum.data=X[i].tostring()#tobeytes if np.version.version >1.9
            datum.label=int(y[i])
            str_id='{:08}'.format(i)
            txn.put(str_id.encode('ascii'),datum.SerializeToString())
train_lmdb_path=path+'mnist_kaggel_train_lmdb'
test_score_lmdb_path=path+'mnist_kaggle_test_lmdb'

# covert_lmdb(X_train,y_train,train_lmdb_path)
# covert_lmdb(X_test,y_test,test_score_lmdb_path)
covert_lmdb(X,y,train_lmdb_path)
covert_lmdb(X,y,test_score_lmdb_path)