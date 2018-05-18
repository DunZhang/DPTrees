# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:28:31 2018

@author: Administrator
"""
from copy import deepcopy
from sklearn import datasets
from DPTools import CartGini,gini,lap,selectByWeight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
class Node:
    def __init__(self,label=None,splitAttr=None,splitPoint=None,isLeaf=True,lNode=None,rNode=None):
        self.label,self.splitAttr,self.splitPoint,self.lNode,self.rNode=label,splitAttr,splitPoint,lNode,rNode
        self.isLeaf=isLeaf

class CARTClassifier:
    def __init__(self,isDisc,epsilon=10,depth=5):
        self.x,self.y,self.attrNumValues,self.isDisc=None,None,[],np.array(isDisc)
        self.xLabens,self.yLaben=[],None
        self.epsilon,self.depth=epsilon,depth
    def __LabelEncoder(self):
        
        newX=np.empty(shape=self.x.shape)
        for i in range(self.x.shape[1]):#对于每一列
            if(self.isDisc[i]):#如果是离散的
                laben=LabelEncoder().fit(self.x[:,i])
                self.xLabens.append(laben)
                self.attrNumValues.append(len(laben.classes_))
                newX[:,i]=laben.transform(self.x[:,i])
            else:
                self.xLabens.append(None)
                self.attrNumValues.append(None)     
                newX[:,i]=self.x[:,i]
        self.x=newX
        self.yLaben=LabelEncoder().fit(self.y)
        self.y=self.yLaben.transform(self.y)
        
    
    def __labelCount(self,y,epsilon):
        di=dict()
        for i in list(y):
            if (i in di.keys()):
                di[i]+=1
            else:
                di[i]=1
        for i in di.keys():
            di[i]+=lap(1,epsilon)
        return di
    def __splitData(self,ins,attrIndex,splitPoint):
        """
            返回分裂后的索引集合
            约定：离散属性第一集合是等于某个值，第二集合是不等于
            连续属性分别是小于等于和大于
        
        """
        res=[[],[]]
        if(self.isDisc[attrIndex]):#是离散属性
            for i in ins:
                if(self.x[i,attrIndex]==splitPoint):
                    res[0].append(i)
                else:
                    res[1].append(i)
        else:#是连续属性
            for i in ins:
                if(self.x[i,attrIndex]<=splitPoint):
                    res[0].append(i)
                else:
                    res[1].append(i)            
        return res
    def dfs(self,node):
        print("是叶节点?:",node.isLeaf,"    分裂属性:",node.splitAttr,"    分裂点:",node.splitPoint,"    叶节点个数:",node.numChildren)
        if(node.isLeaf ==False):
            self.dfs(node.lNode)
            self.dfs(node.rNode)
    def __build(self,ins,attrIns,d,epsilon):
        attrIns=deepcopy(attrIns)
        if(len(attrIns) ==0 or d==self.depth):#属性用尽或者深度已到
            di=self.__labelCount(self.y[ins],epsilon)
            return Node(label=max(di,key=di.get),isLeaf=True)
        else:#是内部结点
            #寻找最佳分裂属性
            e=epsilon/(len(attrIns)+2)
            di=self.__labelCount(self.y[ins],e)
            ginis=np.array(CartGini(self.x[ins,:][:,attrIns],self.y[ins],self.isDisc[attrIns],epsilon=e))
#            minGiniIndex=ginis[:,0].argmin()#attrIns中第maxGainIndex个属性就是最佳分裂属性
            minGiniIndex=selectByWeight(np.exp(ginis[:,0]*(e/4.0)))
            splitAttr=attrIns[minGiniIndex]
            splitPoint=ginis[minGiniIndex,1]
#            minGini=ginis[minGiniIndex,0]
            #根据最佳分类属性将数据集分组
            splitedIns=self.__splitData(ins,splitAttr,splitPoint)
            #将最佳属性从属性列表中移除
            attrIns.remove(splitAttr)
            #根据分组后的数据建立新的结点
            childNodes=[]#childNondes[i]就是该属性第i种值对应的孩子结点
            for i in splitedIns:
#                newNode=node()
                if(len(i)==0):#子节点无数据集可用，将此子节点设置为叶节点
#                    print("子节点无数据",ginis)
                    childNodes.append(Node(label=max(di,key=di.get),isLeaf=True))
                else:
                    childNodes.append(self.__build(i,deepcopy(attrIns),d+1,epsilon))
            return Node(label=max(di,key=di.get),isLeaf=False,splitAttr=splitAttr,splitPoint=splitPoint,
                        lNode=childNodes[0],rNode=childNodes[1]) 
    def fit(self,X,y):
        self.x,self.y=np.array(X),np.array(y)
        self.__LabelEncoder()
        self.root=self.__build(ins=list(range(len(self.x))),attrIns=list(range(self.x.shape[1])),d=0,epsilon=self.epsilon/self.depth)
    def __predict(self,node,x):
#        node=Node()
        if(node.isLeaf):
            return node.label
        else:
            if(self.isDisc[node.splitAttr]):
                if(x[node.splitAttr] == node.splitPoint):
                    return self.__predict(node.lNode,x)
                else:
                    return self.__predict(node.rNode,x)
            else:#连续属性
                if(x[node.splitAttr]<= node.splitPoint):
                    return self.__predict(node.lNode,x)
                else:
                    return self.__predict(node.rNode,x)                
    def predict(self,X):
        res=[]
        x=np.array(X)
        newX=np.empty(shape=x.shape)
        for i in range(x.shape[1]):
            if(self.isDisc[i]):
                newX[:,i]=self.xLabens[i].transform(x[:,i])
            else:
                newX[:,i]=x[:,i]
        for i in range(len(newX)):
            res.append(self.__predict(self.root,newX[i,:]))
        return self.yLaben.inverse_transform(res)
    
if(__name__=="__main__"):
    iris = datasets.load_iris()  # 导入数据集
    X = iris.data # 获得其特征向量
    y = iris.target # 获得样本label
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
    cartC=CARTClassifier(isDisc=[False]*4,depth=3)

    cartC.fit(x_train,y_train) 
    res=cartC.predict(x_test)
    y_test=list(y_test)
    tru=0
    for i in range(len(y_test)):
        if(res[i]==y_test[i]):
            tru+=1
    q=tru/len(y_test)    
     

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    