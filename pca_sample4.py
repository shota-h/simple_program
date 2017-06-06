#自前でPCA 標準パターンをPCA後の平均値とする
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import itertools
import sys

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

            img=cv2.GaussianBlur(img,(5,5),0)
            img = cv2.erode(img,kernel,iterations = 1)
            # ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
            train_img=cv2.GaussianBlur(train_img,(5,5),0)
            train_img = cv2.erode(train_img,kernel,iterations = 1)
            # ret,train_img=cv2.threshold(train_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            train_row[:,i*train_num+j]=train_img.ravel()
            train_class[i*train_num+j]=i
    return train_row,train_class
def func_pca(img_row,n_com):
    m_img=np.mean(img_row,axis=1)
    z=img_row.T-m_img
    cv = np.cov(z.T,bias=1)
    print('cv shape:',cv.shape)
    w, v = np.linalg.eig(cv)
    img_pca=[]
    buff_img=img_row.T.dot(v[:,0:n_com])
    img_pca=np.array(abs(buff_img))
    # img_pca=img_pca.transpose((1,0))
    # print(img_pca.shape)
    temp_pca=[]
    for n in range(class_num):
        mean_p=np.mean(img_pca[n:(n+1)*train_num,:],axis=0)
        temp_pca.append(mean_p)
    temp_pca=np.array(temp_pca)
    # print(temp_pca.shape)
    return img_pca

def knn(train_pca,train_class,n_com,flag_pca):
    score_all=np.zeros([10,1])
    test_row,test_class=Load_test()
    if flag_pca==1:
        test_row=func_pca(test_row,n_com)
    test_row=test_row.T
    # train_pca=train_row[:,:].T.dot(v)
    # print(temp_pca.shape)
    # print(temp_pca.T-train_row[:,0].T)
    print('train_row',train_row.shape)
    print('test_row',test_row.shape)
    for i,j in itertools.product(range(class_num),range(test_num)):
        # norm_train=np.linalg.norm(temp_pca-train_pca[i*train_num+j,:],axis=1)
        norm_train=np.linalg.norm(train_pca-test_row[:,i*test_num+j],axis=1)
        # print(norm_train.shape)
        min_n=np.argmin(norm_train)
        # print(min_n)
        if test_class[i*test_num+j]==train_class[min_n]:
            score_all[i]+=1
    print('score',score_all/test_num)
    print('ave',np.mean(score_all/test_num))

if __name__=='__main__':
    n_com=784
    flag_pca=1
    train_row,train_class=Load_train()
    # print('train',train_row.shape)
    # print(img_row)
    # colors=[plt.cm.jet(float(i)/10,1) for i in range(10)]

    # plt.plot(temp_pca[:,0],temp_pca[:,1],'r.')
    # for n in range(class_num):
    #     plt.scatter(img_pca[n:(n+1)*img_num,0],img_pca[n:(n+1)*img_num,1],c=colors[n],s=20,alpha=0.1,marker='.')
    # plt.show()
    if flag_pca==1:
        train_row=func_pca(train_row,n_com)
        train_row=train_row.T

#ここより上が準備
    knn(train_row.T,train_class,n_com,flag_pca)
    # knn(img_row,img_class,abs(v[:,0:n_com]))
    # knn(temp_pca,abs(v[:,0:n_com]))
