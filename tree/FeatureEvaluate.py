# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:33:26 2018

@author: Administrator
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
def InfoGain(X,y,discrete_features):
    """ 
        discrete_features: 数组类型，与X列相对应，代表指定列是否为离散的
        返回每一列的（信息增益，分裂点，信息增益比）
        X,y:任意形式的数组
        
        注意：离散属性不需要分裂点，为None
    
    """
    X,y=np.array(X),np.array(y)
    discIns,numericIns=[],[]
    #获取X中离散属性和连续属性的索引
    for i in range(len(discrete_features)):
        if(discrete_features[i]):# true represent the feature is discret
            discIns.append(i)
        else:
            numericIns.append(i)
    #分别获取离散属性和连续属性的信息增益值
    if(len(discIns)!=0):
        discRes=__GainDisc(X[:,discIns],y)
    if(len(numericIns)!=0):  
#        print("AA")
        numericRes=__GainNumeric(np.array(X[:,numericIns],dtype=float),y)
    i,j,res=0,0,[]
    #将结果按照X列的顺序放入res中 
    for value in discrete_features:
        if(value):
            res.append(discRes[i])
            i+=1
        else:
            res.append(numericRes[j])
            j+=1            
    return res


def __Ent(y):
    laben=LabelEncoder().fit(y)
    y=laben.transform(y)
    labelNums=len(laben.classes_)
    counts=np.zeros(shape=(labelNums,))
    for value in y: 
        counts[value]+=1
    counts=counts/len(y)
    return (np.log2(counts)*counts*-1.0).sum()


def __GainDisc(X,y):
    res=[]
    
    for i in range(X.shape[1]):
        res.append(__gainDisc(X[:,i],y))
    return res



def __gainDisc(x,y):
    originEnt=__Ent(y)
    laben=LabelEncoder()
    x=laben.fit_transform(x)
    #extract indices
    ins=[]
    for i in range(len(laben.classes_)):
        ins.append([])
        
    for i in range(len(x)):
        ins[x[i]].append(i)
    res=0.0
    #start compute
    iv=0.0
    for i in ins:
        res+=len(i)*__Ent( y[i])
        iv+=len(i)*np.log2(len(i)*1.0/len(x))
    iv/=(-1*len(x))
    return (originEnt-res/len(y),None,(originEnt-res/len(y))/iv)


def __gainNumeric(x,y,sortedIns):
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
    orientGain=__Ent(y)
    # partition y according to poenitial split point (psp)
    maxInfoGain,bestSplitPoint,iv,ivt=float("-inf"),0,0.0,0.0
    for i in range(len(pspis)):
        t=orientGain- ( (pspis[i]+1)*__Ent(y[sortedIns[0:pspis[i]+1]])  +  (len(y)-pspis[i]-1)*__Ent(y[sortedIns[pspis[i]+1:]]) )/len(y)
        if(maxInfoGain<t):
            maxInfoGain=t
            bestSplitPoint=psps[i]
            ivt=i
    iv=( (pspis[ivt]+1)*np.log2((pspis[ivt]+1)*1.0/len(x))  + (len(y)-pspis[ivt]-1)*np.log2((len(y)-pspis[ivt]-1)*1.0/len(x))  )*-1/len(x)
    return(maxInfoGain,bestSplitPoint,maxInfoGain/iv)
def __GainNumeric(X,y):
    res=[]
#    print(X.dtype)
    ins=X.argsort(axis=0)
    for i in range(X.shape[1]):
        res.append(__gainNumeric(X[:,i],y,ins[:,i]))
    return res

def gini(y):
    laben=LabelEncoder().fit(y)
    y=laben.transform(y)
    labelNums=len(laben.classes_)
    counts=np.zeros(shape=(labelNums,))
    for value in y: 
        counts[value]+=1
    counts=counts/len(y)
    return 1-(counts**2).sum()

def __cartGiniNumeric(x,y,sortedIns):
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
    minGini,bestSplitPoint=float("inf"),0
    for i in range(len(pspis)):
        t=( (pspis[i]+1)*gini(y[sortedIns[0:pspis[i]+1]])  +  (len(y)-pspis[i]-1)*gini(y[sortedIns[pspis[i]+1:]]) )/len(y)
        if(minGini>t):
            minGini=t
            bestSplitPoint=psps[i]
    return(minGini,bestSplitPoint)
def __CartGiniNumeric(X,y):
    res=[]
#    print(X.dtype)
    ins=X.argsort(axis=0)
    for i in range(X.shape[1]):
        res.append(__cartGiniNumeric(X[:,i],y,ins[:,i]))
    return res

def __cartGiniDisc(x,y):
    laben=LabelEncoder()
    x=laben.fit_transform(x)    
    ins,insAll=[],set(range(len(x)))
    #initialize
    for i in range(len(laben.classes_)):
        ins.append([])
    #extract indices 
    for i in range(len(x)):
        ins[x[i]].append(i)    
    minGini,index=float("inf"),0
    for i in range(len(ins)):
        #form indices
        Pins,NPins=ins[i],list(insAll-set(ins[i]))
        t=(len(Pins)*gini(y[Pins])  +  len(NPins)*gini(y[NPins]))/len(y)
        if(t<minGini):
            minGini=t
            index=i
    return (minGini,laben.inverse_transform(index))

def __CartGiniDisc(X,y):
    res=[]
    
    for i in range(X.shape[1]):
        res.append(__cartGiniDisc(X[:,i],y))
    return res    
def CartGini(X,y,discrete_features):
    """
    计算cart所用的基尼指数，离散或连续属性都是采用二分的方法
    返回(minGini,splitPoint)
    """
    X,y=np.array(X),np.array(y)
    discIns,numericIns=[],[]
    for i in range(len(discrete_features)):
        if(discrete_features[i]):# true represent the feature is discret
            discIns.append(i)
        else:
            numericIns.append(i)
    discRes=__CartGiniDisc(X[:,discIns],y)
    numericRes=__CartGiniNumeric(np.array(X[:,numericIns],dtype=float),y)
    i,j,res=0,0,[]
    for value in discrete_features:
        if(value):
            res.append(discRes[i])
            i+=1
        else:
            res.append(numericRes[j])
            j+=1            
    return res    
def SETwo(y1,y2):
    y1,y2=np.array(y1),np.array(y2)
    return ((y1-y2)**2).sum()
def se(y):
    return ((y-y.mean())**2).sum()
def __SENumeric(x,y,sortedIns):
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
    minSE,bestSplitPoint=float("inf"),0
    for i in range(len(pspis)):
        t=( (pspis[i]+1)*se(y[sortedIns[0:pspis[i]+1]])  +  (len(y)-pspis[i]-1)*se(y[sortedIns[pspis[i]+1:]]) )/len(y)
        if(minSE>t):
            minSE=t
            bestSplitPoint=psps[i]
    return(minSE,bestSplitPoint)    
def __SEDisc(x,y):
    laben=LabelEncoder()
    x=laben.fit_transform(x)    
    ins,insAll=[],set(range(len(x)))
    #initialize
    for i in range(len(laben.classes_)):
        ins.append([])
    #extract indices 
    for i in range(len(x)):
        ins[x[i]].append(i)    
    minSE,index=float("inf"),0
    for i in range(len(ins)):
        #form indices
        Pins,NPins=ins[i],list(insAll-set(ins[i]))
        t=(len(Pins)*se(y[Pins])  +  len(NPins)*se(y[NPins]))/len(y)
        if(t<minSE):
            minSE=t
            index=i
    return (minSE,laben.inverse_transform(index))
def SE(X,y,discrete_features):
    """
    计算平方误差
    """
    X,y=np.array(X),np.array(y)
    res=[]
    for i in range(len(discrete_features)):
        if(discrete_features[i]):
            
            res.append(__SEDisc(X[:,i],y))
        else:
            x=np.array(X[:,i],dtype=float)
            res.append(__SENumeric(x,y,x.argsort(axis=0)))
    return res


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
