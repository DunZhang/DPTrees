# -*- coding: utf-8 -*-
import random
import math
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
def __sgn(x):
    if (x > 0):
        return 1
    elif (x < 0):
        return -1
    return 0
    
def lap(sensitivity, epsilon):
    """
    返回噪音
    """
    u = random.random() - 0.5;
    return (sensitivity / epsilon) * __sgn(u) * math.log10(1.0 - 2.0 * abs(u));    
    
def selectByWeight(weight):
    """
    根据权重随机选择一个索引,返回该索引
    """
    weight=list(weight)
    s=sum(weight)
    
    w=[x/s for x in weight]
#    print(w)
    randomValue = random.random()
    for i in range(len(w)):
        if (w[i] == 0):
            continue
        if (randomValue < w[i]):
            return i
        else:
            randomValue -= w[i]
    return len(w) - 1
def gini(y):
    laben=LabelEncoder().fit(y)
    y=laben.transform(y)
    labelNums=len(laben.classes_)
    counts=np.zeros(shape=(labelNums,))
    for value in y: 
        counts[value]+=1
    counts=counts/len(y)
    return 1-(1-(counts**2).sum())

def __cartGiniNumeric(x,y,sortedIns,epsilon):
    """
    x:必须ndarray(shape=(n_samples,))
    
    y:必须ndarray(shape=(n_samples,))
    
    sortedIns:list类型或者(n_samples,)
    
    eplison:float   
    
    返回:元组(最佳基尼值和最佳分裂点)
    
    """
    #get poential split points (psps) and poential split point indexs(pspis)
#    sortedIns=list(sortedIns)
    psps,pspis=[],[]
    for i in range(len(sortedIns)-1):
        if(x[sortedIns[i]]!=x[sortedIns[i+1]]):
            psps.append((x[sortedIns[i]]+x[sortedIns[i+1]])/2.0)
            pspis.append(i)
    #pspis[i]:第i个分裂点的索引
    #psps[i]:第i个分裂点的值
    #对于潜在分裂点psps[i],对应划分索引就是 sortedIns[  0:pspis[i]+1  ]
    # partition y according to poenitial split point (psp)
    res=np.empty(shape=(len(pspis),2))#(gini,splitPoint)
    for i in range(len(pspis)):
        res[i,0]=( (pspis[i]+1)*gini(y[sortedIns[0:pspis[i]+1]])  +  (len(y)-pspis[i]-1)*gini(y[sortedIns[pspis[i]+1:]]) )/len(y)
        res[i,1]=psps[i]
    t=selectByWeight(np.exp(res[:,0]*(epsilon/4.0)))
    return(res[t,0],res[t,1])
def __CartGiniNumeric(X,y):
    res=[]
#    print(X.dtype)
    ins=X.argsort(axis=0)
    for i in range(X.shape[1]):
        res.append(__cartGiniNumeric(X[:,i],y,ins[:,i]))
    return res

def __cartGiniDisc(x,y,epsilon):
    """
    x:list或者ndarray(shape=(n_samples,))
    
    y:必须ndarray(shape=(n_samples,))
    
    eplison:float   
    
    返回:元组(最佳基尼值和最佳分裂点)
    
    """
    laben=LabelEncoder()
    x=laben.fit_transform(x)    
    ins,insAll=[],set(range(len(x)))
    #initialize
    for i in range(len(laben.classes_)):
        ins.append([])
    #extract indices 
    for i in range(len(x)):
        ins[x[i]].append(i)    
    res=np.empty(shape=(len(laben.classes_),2))
    for i in range(len(ins)):
        #form indices
        Pins,NPins=ins[i],list(insAll-set(ins[i]))
        res[i,0]=(len(Pins)*gini(y[Pins])  +  len(NPins)*gini(y[NPins]))/len(y)
        res[i,1]=i
    t=selectByWeight(np.exp(res[:,0]*(epsilon/4.0)))
    return (res[t,0],laben.inverse_transform(int(res[t,1])))

def __CartGiniDisc(X,y):
    res=[]
    
    for i in range(X.shape[1]):
        res.append(__cartGiniDisc(X[:,i],y))
    return res    
def CartGini(X,y,discrete_features,epsilon=5.0):
    """
    计算cart所用的基尼指数，离散或连续属性都是采用二分的方法
    返回(minGini,splitPoint)
    """
    X,y=np.array(X),np.array(y)
    res=[]
    for i in range(len(discrete_features)):
        if(discrete_features[i]):# true represent the feature is discret
            res.append(__cartGiniDisc(X[:,i],y,epsilon))
        else:
            x=np.array(X[:,i],dtype=float)
            res.append(__cartGiniNumeric(x,y,x.argsort(axis=0),epsilon))           
    return res    
if (__name__=="__main__"):
    q=CartGini([["A",1.2],["A",3],["A",4],["B",0.5],["B",2.1],["B",10]],["yes","yes","yes","yes","yes","yes"],[True,False])
        
        
        