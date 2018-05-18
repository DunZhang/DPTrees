# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:32:13 2018

@author: Administrator
"""
from copy import deepcopy
from sklearn import datasets
from FeatureEvaluate import SE
from FeatureEvaluate import se
from FeatureEvaluate import SETwo
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
class Node:
    def __init__(self,label=None,splitAttr=None,splitPoint=None,isLeaf=True,numInstances=None,
                 selfSE=None,childSE=None,lNode=None,rNode=None,numChildren=None):
        """
            selfSE:自身作为叶节点时的平方误差
            childSE:孩子的平方误差的加权平均
        
        """
        self.label,self.splitAttr,self.splitPoint,self.lNode,self.rNode=label,splitAttr,splitPoint,lNode,rNode
        self.isLeaf,self.numInstances,self.selfSE,self.childSE=isLeaf,numInstances,selfSE,childSE
        self.numChildren=numChildren

class CARTRegression:
    def __init__(self,isDisc):
        self.x,self.y,self.attrNumValues,self.isDisc=None,None,[],np.array(isDisc)
        self.xLabens=[]
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
#                print(self.x)
                self.xLabens.append(None)
                self.attrNumValues.append(None)     
                newX[:,i]=self.x[:,i]
        self.x=newX
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
    def __CWASE(self,node):
        """
        计算每个节点的孩子的加权平均平方误差
        """
        if(node.isLeaf):#叶子结点：
            node.childSE=node.selfSE
        else:
            node.childSE=(self.__CWASE(node.rNode)*node.rNode.numInstances+self.__CWASE(node.lNode)*node.lNode.numInstances)/node.numInstances
        return node.childSE
    def __numChildren(self,node):
#        node=Node()
        if(node.isLeaf):
            return 1
        else:
            node.numChildren=self.__numChildren(node.lNode)+self.__numChildren(node.rNode)
            return node.numChildren
    def dfs(self,node):
        print("是叶节点?:",node.isLeaf,"  Label:",node.label,"    自身平方差:",node.selfSE,"    孩子平方差加权平均:",node.childSE,
              "    分裂属性:",node.splitAttr,"    分裂点:",node.splitPoint,"    叶节点个数:",node.numChildren)
        if(node.isLeaf ==False):
            self.dfs(node.lNode)
            self.dfs(node.rNode)
    def __build(self,ins,attrIns):
        #是否为根结点
        if(len(attrIns) ==0):#属性用尽
#            print("属性用尽",di)
#            print(len(ins))
            t=se(self.y[ins])
#            print("gini",t)
            return Node(label=self.y[ins].mean(),isLeaf=True,numInstances=len(ins),selfSE=t,childSE=t)
        elif(len(ins)<10):#样本个数比较少
            t=se(self.y[ins])
#            print("gini",t)
            return Node(label=self.y[ins].mean(),isLeaf=True,numInstances=len(ins),selfSE=t,childSE=t)
        else:#是内部结点
            #寻找最佳分裂属性
#            print(self.x[ins,:][:,attrIns])
            ses=np.array(SE(self.x[ins,:][:,attrIns],self.y[ins],self.isDisc[attrIns]))
            minSEIndex=ses[:,0].argmin()#attrIns中第maxGainIndex个属性就是最佳分裂属性
            splitAttr=attrIns[minSEIndex]
            splitPoint=ses[minSEIndex,1]
#            print(ses)
#            minGini=ginis[minGiniIndex,0]
            #根据最佳分类属性将数据集分组
            splitedIns=self.__splitData(ins,splitAttr,splitPoint)
#            print("splitPoint:",splitPoint,"  child set1",len(splitedIns[0]),"  child set2",len(splitedIns[1]))
            #将最佳属性从属性列表中移除
            attrIns.remove(splitAttr)
            #根据分组后的数据建立新的结点
            childNodes=[]#childNondes[i]就是该属性第i种值对应的孩子结点
            for i in splitedIns:
#                newNode=node()
                if(len(i)==0):#子节点无数据集可用，将此子节点设置为叶节点
#                    print("子节点无数据",ginis)
                    childNodes.append(Node(label=self.y[ins].mean(),isLeaf=True,numInstances=0,selfSE=0,childSE=0))
                else:
                    childNodes.append(self.__build(i,deepcopy(attrIns)))
            return Node(label=self.y[ins].mean(),isLeaf=False,splitAttr=splitAttr,splitPoint=splitPoint,
                        lNode=childNodes[0],rNode=childNodes[1],numInstances=len(ins),selfSE=se(self.y[ins])) 
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
        cutNodes,bestSubTree,bestSE=[],-1,float("inf")
        while(True):
            self.__mina=float("inf")
            self.__getACutNode(self.root)
            cutNodes.append(self.__cutNode)
            self.__cutNode.isLeaf=True
            self.__CWASE(self.root)
            t=self.__validationSE(newX,y)
#            t=self.__accuracy(newX,y)
#            print("t:",t)
            if(t<bestSE):
                bestSubTree=len(cutNodes)-1
                bestSE=t
            if(self.__cutNode is self.root):#结束剪枝
                break
        for i in range(bestSubTree+1,len(cutNodes)):
            cutNodes[i].isLeaf=False
#        print(len(cutNodes),bestSubTree)
        self.__CWASE(self.root)
        
    def __getACutNode(self,node):
#        node=Node()
#        global mina,cutNode
        if(node.isLeaf == False):
            t=(node.selfSE-node.childSE)/(node.numChildren-1)
            if(t<self.__mina):
                self.__mina=t
                self.__cutNode=node
            self.__getACutNode(node.lNode)
            self.__getACutNode(node.rNode)
          
    def __validationSE(self,X,y):
        res=[]
        for i in range(len(X)):
            res.append(self.__predict(self.root,X[i,:]))
        return SETwo(res,y)
    
    def fit(self,X,y):

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
        self.x,self.y=np.array(x_train),np.array(y_train)
#        print(self.x)
        self.__LabelEncoder()
#        print(self.x)
        self.root=self.__build(ins=list(range(len(self.x))),
                               attrIns=list(range(self.x.shape[1])))
        self.__CWASE(self.root)
        self.__numChildren(self.root)
#        self.dfs(self.root)
        self.__prune(x_test,y_test)
#        print("减枝后")
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
        return res
    
     

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
