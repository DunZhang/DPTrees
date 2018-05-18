# -*- coding: utf-8 -*-
from FeatureEvaluate import InfoGain
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from copy import deepcopy
class Node:
    def __init__(self,label=None,childNodes=None,splitAttr=None):
        self.label,self.childNodes,self.splitAttr=label,childNodes,splitAttr


class ID3:
    def __init__(self,maxDepth=5):
        self.x,self.y,self.maxDepth,self.attrNumValues=None,None,maxDepth,[]
        self.xLabens,self.yLaben=[],None
    def __LabelEncoder(self):
        
        newX=np.empty(shape=self.x.shape,dtype=int)
        for i in range(self.x.shape[1]):
            laben=LabelEncoder().fit(self.x[:,i])
            self.xLabens.append(laben)
            self.attrNumValues.append(len(laben.classes_))
            newX[:,i]=laben.transform(self.x[:,i])
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
    def __splitData(self,ins,attrIndex):
        res=[]
        for i in range(self.attrNumValues[attrIndex]):
            res.append([])
        for i in ins:
            res[self.x[i,attrIndex]].append(i)
        return res
    def __build(self,ins,attrIns,d):
        #是否为根结点
        di=self.__labelCount(self.y[ins])
#        print(di)
        if(len(di)==1):#数据集只有一种类别的数据
            #叶子结点
#            print(1,list(di.keys())[0])
            return Node(label=list(di.keys())[0])
        elif(len(attrIns) ==0 or d==self.maxDepth):#属性用尽
#            print(2,max(di,key=di.get))
            return Node(label=max(di,key=di.get))
        else:#是内部结点
            #寻找最佳分裂属性
            gains=np.array(InfoGain(self.x[ins,:][:,attrIns],self.y[ins],[True]*len(attrIns)))[:,0]
            maxGainIndex=gains.argmax()#attrIns中第maxGainIndex个属性就是最佳分裂属性
#            maxGain=gains[maxGainIndex]
            splitAttr=attrIns[maxGainIndex]
            #根据最佳分类属性将数据集分组
            splitedIns=self.__splitData(ins,splitAttr)
            #将最佳属性从属性列表中移除
            attrIns.remove(splitAttr)
            #根据分组后的数据建立新的结点
            childNodes=[]#childNondes[i]就是该属性第i种值对应的孩子结点
            for i in splitedIns:
#                newNode=node()
                if(len(i)==0):#子节点无数据集可用，将此子节点设置为叶节点
#                    print(max(di,key=di.get))
                    childNodes.append(Node(label=max(di,key=di.get)))
                else:
                    childNodes.append(self.__build(i,deepcopy(attrIns),d+1))
            return Node(label=None,childNodes=childNodes,splitAttr=splitAttr)
    def fit(self,X,y):
        self.x,self.y=np.array(X),np.array(y)
        self.__LabelEncoder()
        self.root=self.__build(ins=list(range(len(self.x))),
                               attrIns=list(range(self.x.shape[1])),
                               d=0)
    def __predict(self,node,x):
#        node=Node()
        if(node.label is None):
            return self.__predict(node.childNodes[x[node.splitAttr]],x)
        else:
#            print(node.label)
#            return self.yLaben.inverse_transform(node.label)
            return node.label
    def predict(self,x):
        res=[]
        x=np.array(x)
        newX=np.empty(shape=x.shape,dtype=int)
        for i in range(x.shape[1]):
            newX[:,i]=self.xLabens[i].transform(x[:,i])
        for i in range(len(newX)):
            res.append(self.__predict(self.root,newX[i,:]))
        return self.yLaben.inverse_transform(res)
        
        
                

    
    
    
    
    
    
    
    
    
    