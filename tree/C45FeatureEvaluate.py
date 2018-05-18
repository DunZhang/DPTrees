# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from FeatureEvaluate import InfoGain
def C45Gain(weight,X,y,isDisc):
    """
    返回（信息增益，分裂点，信息增益比）
    """
    x,y,weight=np.array(X),np.array(y),np.array(weight)  
    #find out the instances without missing values for each column
    normalIns=[]
    for i in range(x.shape[1]):
        normalIns.append([])
        for j in range(x.shape[0]):
            if(str(x[j,i])!='?'):
                normalIns[-1].append(j)
    
    gains=[]
    for i in range(x.shape[1]):
        gains.append(   InfoGain(x[normalIns[i],i].reshape((len(normalIns[i]),1)),y[normalIns[i]],[isDisc[i]])[0] )
        
    gains=np.array(gains)    
    for i in range(x.shape[1]):
        p=weight[normalIns[i]].sum()/weight.sum()
#        print(p)
        gains[i,0]*=p
        gains[i,2]*=p   
    return gains
    
    

if (__name__=='__main__'):
#    data=pd.read_excel("D:/xg.xlsx")
#    x,y=data.iloc[:,0:-1],data.iloc[:,-1]
#    weight=[1.0]*len(x)
#    res=C45Gain(weight,x,y,[True,True,True,True,True,True,False,False])
#    res=pd.DataFrame(res)
    pass
    