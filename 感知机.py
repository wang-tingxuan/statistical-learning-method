import numpy as np
import time
from sklearn.linear_model import Perceptron
x=np.asarray([[3,3],[4,3],[1,1]])
y=np.asarray([1,1,-1])

class Ganzhiji():
    def chushihua(self,x,n1):
        self.g=np.dot(x,x.transpose((1,0)))
        self.a=np.zeros(3)
        self.b=0
        self.w=np.zeros(2)
        self.n1=n1
    def duiou(self,x,y):      
        while True:
            flag=True
            for i in range(x.shape[0]):
                if y[i]*(np.dot(self.a*y,self.g[i])+self.b)<=0:
                    self.a[i]=self.a[i]+self.n1
                    self.b=self.b+self.n1*y[i]
                    flag=False
                    break
            if flag:
                break
        self.w=np.dot(self.a*y,x)
    def display(self):
        print("分割超平面为："+str(self.w[0])+'x1+ '+str(self.w[1])+'x2+ '+str(self.b))
        
    def yuanshi(self,x,y):
        while True:
            flag=True
            for i in range(x.shape[0]):
                if y[i]*(np.dot(self.w,x[i])+self.b)<=0:
                    self.w= self.w+self.n1*y[i]*x[i]
                    self.b= self.b+self.n1*y[i]
                    flag=False
                    break
            if flag:
                break
    def model_of_sklearn(self,x,y):
        perceptron=Perceptron()
        perceptron.fit(x,y)
        print("w:",perceptron.coef_,"\n","b:",perceptron.intercept_,"\n")
    def time_test(self):
        start=time.time()
        for i in range(100000):
            ganzhiji.chushihua(x,1)
            ganzhiji.yuanshi(x,y)
        end=time.time()
        print('原始形式100000次时间： '+str(end-start))
        
        start=time.time()
        for i in range(100000):
            ganzhiji.chushihua(x,1)
            ganzhiji.duiou(x,y)
        end=time.time()
        print('对偶形式100000次时间： '+str(end-start))
ganzhiji=Ganzhiji()
print("对偶形式")
for i in range(10):      
    per = np.random.permutation(x.shape[0])
    x1 = x[per]
    y1 = y[per]
    ganzhiji.chushihua(x1,1)
    ganzhiji.duiou(x1,y1)
    ganzhiji.display()
print('.....................................................')
print('原始形式')

for i in np.linspace(0.1,1,10):
    ganzhiji.chushihua(x,i)
    ganzhiji.yuanshi(x,y)
    ganzhiji.display()
    
print('.....................................................')
print('sklearn.linear_model 的Perceptron模块')
ganzhiji.chushihua(x,1)
ganzhiji.model_of_sklearn(x,y)
    
ganzhiji.time_test()