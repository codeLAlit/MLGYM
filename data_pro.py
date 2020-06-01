import numpy as np
import pandas as pd
pd.set_option('display.max_columns',15)

data_train=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')
print(data_train.info())
print(data_train.columns)
(m,n)=data_train.shape
for col in data_train.columns:
    na_count=data_train[col].isna().sum()
    if(na_count>=0.50*m):
        data_train.drop(columns=[col],inplace=True)
        data_test.drop(columns=[col],inplace=True)
        n-=1
#all columns with data that have more than 10% unique values and they being strings they don't have any significance
#in the data
for col in data_train.columns:
    uniq=len(data_train[col].unique())
    if(uniq>0.1*m):
        if(data_train[col].dtype=='object'):
            data_train.drop(columns=[col],inplace=True)
            data_test.drop(columns=[col],inplace=True)
            n-=1

#do this only incase we are allowing users to input one csv file with both result column also included
#first we should somehow get the name of column( case sensitive )  that contains result let this be res_name
#here res name='Survived'
res_name='Survived'
res_data=data_train[res_name].copy()
data_train.drop(columns=res_name,inplace=True)
n-=1
for col in data_train.columns:
    if(data_train[col].dtype!='object'):
        data_train[col].fillna(data_train[col].mean(),inplace=True)
        data_test[col].fillna(data_test[col].mean(),inplace=True)
    else:
        data_train[col].fillna(data_train[col].describe().top,inplace=True)
        data_test[col].fillna(data_test[col].describe().top,inplace=True)
for col in data_train.columns:
    if(data_train[col].dtype!='object'):
        pass
    else:
        mems=data_train[col].unique()
        dictionary={}
        i=0
        for mem in mems:
            dictionary[mem]=i
            i+=1
        headings=[]
        for j in range(len(mems)):
            headings.append(col+'_'+mems[j])
        data_train[col]=data_train[col].map(dictionary)
        data_test[col]=data_test[col].map(dictionary)
        temp_train=np.zeros((data_train.shape[0],len(dictionary)))
        temp_test=np.zeros((data_test.shape[0],len(dictionary)))
        
        temp_train[np.arange(data_train.shape[0]),data_train[col]]=1
        temp_test[np.arange(data_test.shape[0]),data_test[col]]=1
        tempdf_train=pd.DataFrame(temp_train,columns=headings)
        tempdf_test=pd.DataFrame(temp_test,columns=headings)
        data_train=data_train.join(tempdf_train)
        data_test=data_test.join(tempdf_test)
        data_train.drop(columns=col,inplace=True)
        data_test.drop(columns=col,inplace=True)
var1=data_train.mean()
var2=data_train.std()
data_train=(data_train-var1)/var2
data_test=(data_test-var1)/var2




