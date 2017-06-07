#初期代表パターンを各クラスの平均値
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import itertools
import sys
from collections import defaultdict


file_ex='.PGM'
class_num=10
train_num=200
img_pix=28
test_num=1000
# img=np.zeros((28,28,class_num*img_num))
train_row=np.zeros((img_pix*img_pix,class_num*train_num))
test_row=np.zeros((img_pix*img_pix,class_num*test_num))
train_class=np.zeros((class_num*train_num,1))
test_class=np.zeros((class_num*test_num,1))
kernel=np.ones((3,3),np.uint8)
kernwl=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)


def Load_test():
    for i in range(0,class_num):
        for  j in range(0,test_num):
            jj='{0:04d}'.format(j)
            img_name=str(i)+'-'+str(jj)+file_ex
            img=cv2.imread('C:/Users/admin/programe/programe/pattern_recognition/Images/TestSamples/'+img_name,0)
            if img is None:
                print(img_name)
                print('not read')
                sys.exit()

            # img=cv2.GaussianBlur(img,(5,5),0)
            # img = cv2.erode(img,kernel,iterations = 1)
            ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            test_row[:,i*test_num+j]=img.ravel()
            test_class[i*test_num+j]=i
    return test_row,test_class


def Load_train():
    for i in range(0,class_num):
        for  j in range(0,train_num):
            jj='{0:04d}'.format(j)
            img_name=str(i)+'-'+str(jj)+file_ex
            train_img=cv2.imread('C:/Users/admin/programe/programe/pattern_recognition/Images/TrainingSamples/'+img_name,0)
            if train_img is None:
                print(img_name)
                print('not read')
                sys.exit()
            # train_img=cv2.GaussianBlur(train_img,(5,5),0)
            # train_img = cv2.erode(train_img,kernel,iterations = 1)

            train_row[:,i*train_num+j]=train_img.ravel()
            train_class[i*train_num+j]=i
    return train_row,train_class

def func_lvq(m_train,train_row):
    alpha=0.01
    for i,j in itertools.product(range(class_num),range(train_num)):


if __name__=='__main__':
    train_row,train_class=Load_train()
    m_train=[]
    m_class=np.array(range(class_num))
    for i in range(class_num):
        buff_mean=np.mean(train_row[:,i*train_num:(i+1)*train_num],axis=1)
        m_train.append(buff_mean)
    m_train=np.array(m_train)
