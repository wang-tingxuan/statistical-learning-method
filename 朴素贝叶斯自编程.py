# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
X = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']]
X1 = list(zip(X[0],X[1]))
Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
XY = list(zip(X1,Y))
XY.sort(key=lambda x:x[-1])
N = np.shape(XY)[0]          #实例数
n= len(X)                    #特征数
k = len(set(Y))              #标签数
x = [2, 'S']
lamb = 0.2

s=[]                        #每种特征的类别数
for xx in X:
    s.append(len(set(xx)))
prior={}
posterior={}
dictxy={}
from itertools import groupby
for i,group in groupby(XY, key=lambda x:x[-1]):
    dictxy[i]=list(group)
    prior[i]=(len(dictxy[i])+lamb) /(N+k*lamb)
    posterior[i]=1.
    
for key in dictxy.keys():
    for n1 in range(n):
        count=0
        for sample in dictxy[key]:    
            if x[n1]== sample[0][n1]:
                count+=1
        posterior[key]*= (count+lamb)/(len(dictxy[key])+s[n1]*lamb)
    posterior[key]*=prior[key]
label=sorted(posterior.items(), key=lambda x:x[-1],reverse=True)[0][0]
print('后验概率为:',posterior)
print('样本',x,'的预测类别为:',label )