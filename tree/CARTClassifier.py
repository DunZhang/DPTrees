# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:28:31 2018

@author: Administrator
"""
from copy import deepcopy
from sklearn import datasets
from FeatureEvaluate import CartGini
from FeatureEvaluate import gini
from FeatureEvaluate import SE
from FeatureEvaluate import SETwo
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
class Node:
    def __init__(self,label=None,splitAttr=None,splitPoint=None,isLeaf=True,numInstances=None,
                 selfGini=None,childWGinis=None,lNode=None,rNode=None,numChildren=None):
        self.label,self.splitAttr,self.splitPoint,self.lNode,self.rNode=label,splitAttr,splitPoint,lNode,rNode
        self.isLeaf,self.numInstances,self.selfGini,self.childWGinis=isLeaf,numInstances,selfGini,childWGinis
        self.numChildren=numChildren

class CARTClassifier:
    def __init__(self,isDisc):
        self.x,self.y,self.attrNumValues,self.isDisc=None,None,[],np.array(isDisc)
        self.xLabens,self.yLaben=[],None
        self.__mina,self.__cutNode=None,None
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
        
    
    def __labelCount(self,y):
        di=dict()
        for i in list(y):
            if (i in di.keys()):
                di[i]+=1
            else:
                di[i]=1
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
    def __CWAG(self,node):
        """
        计算每个节点的孩子的加权平均基尼指数
        """
        if(node.isLeaf):#叶子结点：
            node.childWGinis=node.selfGini           
        else:
            node.childWGinis=(self.__CWAG(node.rNode)*node.rNode.numInstances+self.__CWAG(node.lNode)*node.lNode.numInstances)/node.numInstances
        return node.childWGinis
    def __numChildren(self,node):
#        node=Node()
        if(node.isLeaf):
            return 1
        else:
            node.numChildren=self.__numChildren(node.lNode)+self.__numChildren(node.rNode)
            return node.numChildren
    def dfs(self,node):
        print("是叶节点?:",node.isLeaf,"    自身基尼:",node.selfGini,"    孩子基尼加权平均:",node.childWGinis,
              "    分裂属性:",node.splitAttr,"    分裂点:",node.splitPoint,"    叶节点个数:",node.numChildren)
        if(node.isLeaf ==False):
            self.dfs(node.lNode)
            self.dfs(node.rNode)
    def __build(self,ins,attrIns):
        #是否为根结点
        di=self.__labelCount(self.y[ins])
#        print(di)
        if(len(di)==1):#数据集只有一种类别的数据
            #叶子结点
#            print("只有一种类别",di)
            return Node(label=list(di.keys())[0],isLeaf=True,numInstances=len(ins),selfGini=0,childWGinis=0)
        elif(len(attrIns) ==0):#属性用尽
#            print("属性用尽",di)
#            print(len(ins))
            t=gini(self.y[ins])
#            print("gini",t)
            return Node(label=max(di,key=di.get),isLeaf=True,numInstances=len(ins),selfGini=t,childWGinis=t)
        else:#是内部结点
            #寻找最佳分裂属性
            ginis=np.array(CartGini(self.x[ins,:][:,attrIns],self.y[ins],self.isDisc[attrIns]))
            minGiniIndex=ginis[:,0].argmin()#attrIns中第maxGainIndex个属性就是最佳分裂属性
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
                    childNodes.append(Node(label=max(di,key=di.get),isLeaf=True,numInstances=0,selfGini=0,childWGinis=0))
                else:
                    childNodes.append(self.__build(i,deepcopy(attrIns)))
            return Node(label=max(di,key=di.get),isLeaf=False,splitAttr=splitAttr,splitPoint=splitPoint,
                        lNode=childNodes[0],rNode=childNodes[1],numInstances=len(ins),selfGini=gini(self.y[ins])) 
    def __prune(self,X,y):
        """
        后剪枝
        """
        
        x=np.array(X)
        newX=np.empty(shape=x.shape)
        for i in range(x.shape[1]):
            if(self.isDisc[i]):
                newX[:,i]=self.xLabens[i].transform(x[:,i])
            else:
                newX[:,i]=x[:,i]
        cutNodes,bestSubTree,bestGini=[],-1,float("-inf")
        while(True):
            self.__mina=float("inf")
            self.__getACutNode(self.root)
            cutNodes.append(self.__cutNode)
            self.__cutNode.isLeaf=True
            self.__CWAG(self.root)
#            t=self.__validationGini(newX,y)
            t=self.__accuracy(newX,y)
#            print("t:",t)
            if(t>bestGini):
                bestSubTree=len(cutNodes)-1
                bestGini=t
            if(self.__cutNode is self.root):#结束剪枝
                break
        for i in range(bestSubTree+1,len(cutNodes)):
            cutNodes[i].isLeaf=False
        print(len(cutNodes),bestSubTree)
        self.__CWAG(self.root)
        
    def __getACutNode(self,node):
#        node=Node()
#        global mina,cutNode
        if(node.isLeaf == False):
            t=(node.selfGini-node.childWGinis)/(node.numChildren-1)
            if(t<self.__mina):
                self.__mina=t
                self.__cutNode=node
            self.__getACutNode(node.lNode)
            self.__getACutNode(node.rNode)
        
    def __accuracy(self,X,y):
        res=[]
        for i in range(len(X)):
            res.append(self.__predict(self.root,X[i,:]))
        res=list( self.yLaben.inverse_transform(res))                    
        y=list(y)
        tru=0
        for i in range(len(y)):
            if(res[i]==y[i]):
                tru+=1
        return tru/len(y)        
    def __validationGini(self,X,y):
        res=[]
        for i in range(len(X)):
            res.append(self.__predict(self.root,X[i,:]))
        res=list( res)
        ins=dict()                    
        y=np.array(y)
        for i in range(len(res)):
            if(res[i] not in ins.keys()):
                ins[res[i]]=[]
                ins[res[i]].append(i)
            else:
                ins[res[i]].append(i)
        t=0.0
        for i in ins.keys():
            t+=len(ins[i])*gini(y[ins[i]])
        return t/len(res)
    
    def fit(self,X,y):

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10,stratify=y)
        self.x,self.y=np.array(x_train),np.array(y_train)
        self.__LabelEncoder()
        self.root=self.__build(ins=list(range(len(self.x))),
                               attrIns=list(range(self.x.shape[1])))
        self.__CWAG(self.root)
        self.__numChildren(self.root)
#        self.dfs(self.root)
        self.__prune(x_test,y_test)
#        print("减值后")
#        self.dfs(self.root)
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
    
     

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    