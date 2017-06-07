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
kernel=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)


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

            # img=cv2.GaussianBlur(img,(3,3),0)
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
            # train_img=cv2.GaussianBlur(train_img,(3,3),0)
            # train_img = cv2.erode(train_img,kernel,iterations = 1)

            train_row[:,i*train_num+j]=train_img.ravel()
            train_class[i*train_num+j]=i
    return train_row,train_class

def func_lvq(mean_train,train_row,alpha):
    m_train=np.copy(mean_train)

    for i,j in itertools.product(range(class_num),range(train_num)):
        diff_train=m_train-train_row[i*train_num+j,:]
        norm_train=np.linalg.norm(diff_train,axis=1)
        cn=np.argmin(norm_train)
        # print(cn)
        if cn == train_class[i*train_num+j]:
            m_train[cn,:]+=alpha*diff_train[cn,:]
        else:
            m_train[cn,:]-=alpha*diff_train[cn,:]

    return m_train

def knn(temp_row):
    score_all=np.zeros([10,1])
    test_row,test_class=Load_test()
    mat_ans=np.zeros([10,10])
    for i,j in itertools.product(range(class_num),range(test_num)):
        diff_test=temp_row-test_row.T[i*test_num+j,:]
        norm_train=np.linalg.norm(diff_test,axis=1)
        min_n=np.argmin(norm_train)
        mat_ans[i,min_n]+=1
        if test_class[i*test_num+j]==min_n:
            score_all[i]+=1
    # print('score\n',score_all/test_num)
    # print('ave:',np.mean(score_all/test_num*100),'%')
    return np.mean(score_all/test_num*100)
    # print(mat_ans)


if __name__=='__main__':
    alpha=1
    da=alpha*10000
    train_row,train_class=Load_train()
    m_train=[]
    m_class=np.array(range(class_num))
    # print(train_row.shape)
    for i in range(class_num):
        buff_mean=np.mean(train_row[:,i*train_num:(i+1)*train_num],axis=1)
        m_train.append(buff_mean)
    m_train=np.array(m_train)
    # m=np.copy(m_train)
    i=0
    vec_ave=[]
    while i<100:
        # print(str(i),alpha)
        m_train1=func_lvq(m_train,train_row.T,alpha)
        buff_ave=knn(m_train1)
        vec_ave.append(buff_ave)
        alpha/=2
        i+=1
    vec_ave=np.array(vec_ave)
    print(np.max(vec_ave),0.5**np.argmax(vec_ave))
