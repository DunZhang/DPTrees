# -*- coding: utf-8 -*-

from C45FeatureEvaluate import C45Gain
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from copy import deepcopy
class Node:
    def __init__(self,label=None,splitAttr=None,splitPoint=None,isLeaf=True,numInstances=None,
                 childNodes=None,numLeaf=None,numMC=None,numChildMC=None):
        self.label,self.splitAttr,self.splitPoint,self.childNodes=label,splitAttr,splitPoint,childNodes
        self.isLeaf,self.numInstances=isLeaf,numInstances
        self.numLeaf,self.numMC,self.numChildMC=numLeaf,numMC,numChildMC

class C45:
    def __init__(self,isDisc):
        self.x,self.y,self.attrNumValues,self.isDisc=None,None,[],np.array(isDisc)
        self.yLaben,self.xLabels=None,None
        self.__mina,self.__cutNode=None,None
    def __LabelEncoder(self):
        self.yLaben=LabelEncoder().fit(self.y)
        self.y=self.yLaben.transform(self.y)
        self.xLabels=[]
        for i in range(self.x.shape[1]):
            if(self.isDisc[i]):
                t=set([str(x) for x in self.x[:,i]])
                if("?" in t):
                    t.remove('?')
                self.xLabels.append( list(t) )
            else:
                self.xLabels.append(None)
            
        
    
    def __labelCount(self,y):
        di=dict()
        for i in list(y):
            if (i in di.keys()):
                di[i]+=1
            else:
                di[i]=1
        return di
    def __splitData(self,ins,attrIndex,splitPoint,weight):
        """
            返回分裂后的索引集合
            每一块索引分为正常的索引和缺失索引的集合
            约定：连续属性分别是小于等于和大于
        
        """   
        missingIns,missingWeight,sumWeight=[],[],0.0
        resIns,resWeight={},{}
        if(self.isDisc[attrIndex]):#是离散属性
            #先加入正常的索引                        
            for w,i in zip(weight,ins):
                value=str(self.x[i,attrIndex])
                if(value=="?"):#缺失值
#                    print("处理了缺失值")
                    missingIns.append(i)
                    missingWeight.append(w)
                else:
                    sumWeight+=w
                    if(value  in resIns.keys() ):
                        resIns[value].append(i)
                        resWeight[value].append(w)
                    else:
                        resIns[value]=[i]
                        resWeight[value]=[w]
            #再加入缺失值的索引
            for key in resIns.keys():
                resIns[key]+=missingIns
                rv=sum(resWeight[key])/sumWeight
                resWeight[key]+=[e*rv for e in missingWeight]
        else:#是连续属性
            resIns,resWeight={0:[],1:[]},{0:[],1:[]}
            for w,i in zip(weight,ins):
                value=float(self.x[i,attrIndex])
                if(self.x[i,attrIndex]<=splitPoint):
                    resIns[0].append(i)
                    resWeight[0].append(w)
                else:
                    resIns[1].append(i)  
                    resWeight[1].append(w)
        return (resIns,resWeight)
    def __numLeaf(self,node):
        """
            计算每个结点的叶节点个数
        """
#        node=Node()
        if(node.isLeaf):
            return 1
        else:
            t=0.0
            for key in node.childNodes.keys():
                t+=self.__numLeaf(node.childNodes[key])
            node.numLeaf=t
            return node.numLeaf
    def __numChildMC(self,node):
#        node=Node()
        if(node.isLeaf):
            return node.numMC
        else:
            t=0.0
            for key in node.childNodes.keys():
                t+=self.__numChildMC(node.childNodes[key])
            node.numChildMC=t
            return node.numChildMC
        
    def __prune(self,node):
#        node=Node()
        if(node.isLeaf):
            return
        nodeMC=node.numMC+0.5#该结点的误分类个数的修正值
        nodeChildMC=node.numChildMC+0.5*node.numLeaf#该节点的孩子结点中误分类个数的修正值
        se=(( nodeChildMC*(node.numInstances - nodeChildMC)  )/node.numInstances)**0.5
        if(nodeMC<=nodeChildMC+se):
            print("cut")
            node.isLeaf=True
#            for key in node.childNodes.keys():               
#                del node.childNodes[key]            
        else:
            for key in node.childNodes.keys():               
                self.__prune(node.childNodes[key])
        
    def dfs(self,node):
        print("是叶节点?:",node.isLeaf,"    分裂属性:",node.splitAttr,"    分裂点:",node.splitPoint,"  实例个数:",node.numInstances)
        if(node.isLeaf ==False):
            for key in node.childNodes.keys():
                self.dfs(node.childNodes[key])
            
            
            
    def __build(self,ins,attrIns,weight):
        #是否为根结点
        di=self.__labelCount(self.y[ins])
        label=max(di,key=di.get)
        numMC=0.0
        for i in di.keys():
            if(i!=label):
                numMC+=di[i]
#        print(di)
        if(len(di)==1):#数据集只有一种类别的数据
            #叶子结点
#            print("只有一种类别",di)
            return Node(label=label,isLeaf=True,numInstances=len(ins),numMC=0.0)
        elif(len(attrIns) ==0):#属性用尽
#            print("属性用尽",di)
#            print(len(ins))
#            t=gini(self.y[ins])
#            print("gini",t)
            return Node(label=label,isLeaf=True,numInstances=len(ins),numMC=numMC)
        else:#是内部结点
            #寻找最佳分裂属性
            grs=np.array(C45Gain(weight,self.x[ins,:][:,attrIns],self.y[ins],self.isDisc[attrIns]))
            maxGRIndex=grs[:,2].argmax()#attrIns中第maxGainIndex个属性就是最佳分裂属性
            splitAttr=attrIns[maxGRIndex]
            splitPoint=grs[maxGRIndex,1]
#            minGini=ginis[minGiniIndex,0]
            #根据最佳分类属性将数据集分组
            splitedIns,splitedWeight=self.__splitData(ins,splitAttr,splitPoint,weight)
            #将最佳属性从属性列表中移除,只移除离散属性
            if(self.isDisc[splitAttr]):
                attrIns.remove(splitAttr)
            #根据分组后的数据建立新的结点
            childNodes={}
            if(self.isDisc[splitAttr]):
                for key in self.xLabels[splitAttr]:
    #                newNode=node()
                    if(key not in splitedIns.keys()):#子节点无数据集可用，将此子节点设置为叶节点
    #                    print("子节点无数据",ginis)
                        childNodes[key]=Node(label=label,isLeaf=True,numInstances=0,numMC=0.0)
                    else:
                        childNodes[key]=self.__build(splitedIns[key],deepcopy(attrIns),splitedWeight[key])
            else:
                childNodes[0]=self.__build(splitedIns[0],deepcopy(attrIns),splitedWeight[0])
                childNodes[1]=self.__build(splitedIns[1],deepcopy(attrIns),splitedWeight[1])
            
            return Node(label=label,isLeaf=False,splitAttr=splitAttr,splitPoint=splitPoint,
                        childNodes=childNodes,numInstances=len(ins),numMC=numMC) 
#    def __prune(self,X,y):
#        """
#        后剪枝
#        """
#        
#        x=np.array(X)
#        newX=np.empty(shape=x.shape)
#        for i in range(x.shape[1]):
#            if(self.isDisc[i]):
#                newX[:,i]=self.xLabens[i].transform(x[:,i])
#            else:
#                newX[:,i]=x[:,i]
#        cutNodes,bestSubTree,bestGini=[],-1,float("-inf")
#        while(True):
#            self.__mina=float("inf")
#            self.__getACutNode(self.root)
#            cutNodes.append(self.__cutNode)
#            self.__cutNode.isLeaf=True
#            self.__CWAG(self.root)
##            t=self.__validationGini(newX,y)
#            t=self.__accuracy(newX,y)
##            print("t:",t)
#            if(t>bestGini):
#                bestSubTree=len(cutNodes)-1
#                bestGini=t
#            if(self.__cutNode is self.root):#结束剪枝
#                break
#        for i in range(bestSubTree+1,len(cutNodes)):
#            cutNodes[i].isLeaf=False
#        print(len(cutNodes),bestSubTree)
#        self.__CWAG(self.root)

    def fit(self,X,y):
        self.x,self.y=np.array(X),np.array(y)
        self.__LabelEncoder()
        self.root=self.__build(ins=list(range(len(self.x))),
                               attrIns=list(range(self.x.shape[1])),
                               weight=[1.0]*len(self.x))
#        self.dfs(self.root)
        self.__numLeaf(self.root)
        self.__numChildMC(self.root)
        self.__prune(self.root)
        self.dfs(self.root)
    def __predict(self,node,x):
        node=Node()
        if(node.isLeaf):
            return node.label
        else:
            if(self.isDisc[node.splitAttr]):
                return self.__predict(node.childNodes[x[node.splitAttr]],x)
            else:#连续属性
                if(float(x[node.splitAttr])<= node.splitPoint):
                    return self.__predict(node.childNodes[0],x)
                else:
                    return self.__predict(node.childNodes[1],x)                
#    def predict(self,X):
#        res=[]
#        x=np.array(X)
#        newX=np.empty(shape=x.shape)
#        for i in range(x.shape[1]):
#            if(self.isDisc[i]):
#                newX[:,i]=self.xLabens[i].transform(x[:,i])
#            else:
#                newX[:,i]=x[:,i]
#        for i in range(len(newX)):
#            res.append(self.__predict(self.root,newX[i,:]))
#        return self.yLaben.inverse_transform(res)
    def predict(self,X):
        res=[]
        x=np.array(X)
        for i in range(len(x)):
            res.append(self.__predict(self.root,x[i,:]))
        return self.yLaben.inverse_transform(res)    
if(__name__=="__main__"):
#    data=pd.read_csv("D:/car.txt",sep=',',header=None)
#    data.iloc[0,0]="?"
#    data.iloc[0,3]="?"
#    data.iloc[1,2]="?"
#    data.iloc[3,1]="?"
#    data.iloc[23,4]="?"
#    data.iloc[13,5]="?"
#    x,y=data.iloc[:,0:-1],data.iloc[:,-1]
#    c45=C45([True]*6)
#    c45.fit(x,y)
    pass
#    print(list(di.keys())[0])
#    print(max(di,key=di.get))
    
























