# ML-4-Logic-Regression-with-regularization
Use regularization to handle overfitting in logic regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#数据预处理
path="ex2data2.txt"
data=pd.read_csv(path,header=None,names=['Test1','Test2','Accepted'])
print(data.head())

#data.insert(0,'Ones',1)
#cols=data.shape[1]
#X=data.iloc[:,0:cols-1]
#y=data.iloc[:,cols-1:cols]
#X=np.array(X.values)
#y=np.array(y.values)
#theta=np.zeros([cols-1])
#print(X.shape,y.shape,theta.shape)

#创建散点图，可视化
positive=data[data['Accepted'].isin([1])]
negative=data[data['Accepted'].isin([0])]

fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(positive['Test1'],positive['Test2'],s=50,c='b',marker='o',label='Accepted')
ax.scatter(negative['Test1'],negative['Test2'],s=50,c='r',marker='x',label='Rejected')
ax.legend()
ax.set_xlabel('Test1 score')
ax.set_ylabel('Test2 score')
plt.show()
                 
#特征映射
def mapfeature(x1,x2,power,as_ndarray=False):
    data={}
    for i in np.arange(power+1):
        for p in np.arange(i+1):
            data["f{}{}".format(i-p,p)]=np.power(x1,i-p)*np.power(x2,p)
    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)
x1=np.array(data.Test1)
x2=np.array(data.Test2)
X=mapfeature(x1,x2,6)
print(X.head())
X=np.array(X.values)
cols=data.shape[1]
#X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
#y=data.Accepted
y=np.array(y.values)
theta=np.zeros([X.shape[1]])
print(X.shape,y.shape,theta.shape)
#定义sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#定义正则化的代价函数和梯度下降
def costfunction(theta,X,y,L=1):
    m=len(X)
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=(np.matrix(y))
    first=y.T*(np.log(sigmoid(X*theta.T)))
    second=(1-y).T*(np.log(1-sigmoid(X*theta.T)))
    theta_j1to_n=theta[1:]
    regularized_term=(1/(2*m))*np.power(theta_j1to_n,2).sum()
    J=-(first+second)/m+regularized_term
    return J
print(costfunction(theta,X,y,L=1))

def gradient(theta,X,y,L=1):
    m=len(X)
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=(np.matrix(y))
    #grad=np.zeros(3)
    #inner1=(sigmoid(X*theta.T)-y).T*X
    #grad=inner1/m
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    error=sigmoid(X*theta.T)-y
    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        if i==0:
            grad[i]=np.sum(term)/m
        else:
            grad[i]=np.sum(term)/m+(1/m)*theta[:,i]
            
    return grad
print(gradient(theta,X,y,L=1))
#求解器qiujie
import scipy.optimize as opt
#result=opt.fmin_tnc(func=costfunction,x0=theta,fprime=gradient,args=(X,y))
result=opt.minimize(fun=costfunction,x0=theta,args=(X,y),method='Newton-CG',jac=gradient)
print(result)

#定义预测函数模型并分析结果
def predict(theta,X):
    probability=sigmoid(X@theta.T)
    return[1 if x>=0.5 else 0 for x in probability] 
from sklearn.metrics import classification_report  
theta_min=result.x
p=predict(theta_min,X)
#print(p)
print(classification_report(y,p))
#绘制边界
def draw_boundary(power, l):
#     """
#     power: polynomial power for mapped feature
#     l: lambda constant
#     """
    density = 1000
    threshhold = 2 * 10**-3
    final_theta =result.x
    x, y = find_decision_boundary(density, power, final_theta, threshhold)
    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot('test1', 'test2', hue='accepted', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})
    plt.scatter(x, y, c='R', s=10)
    plt.title('Decision boundary')
    plt.show()

def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = mapfeature(x_cord, y_cord, power)  # this is a dataframe
    inner_product = mapped_cord.as_matrix() @ theta
    decision = mapped_cord[np.abs(inner_product) < threshhold]
    return decision.f10, decision.f01
#寻找决策边界函数
draw_boundary(power=6, l=1)

