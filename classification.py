import opendatasets as od
od.download("https://www.kaggle.com/competitions/titanic/data?select=train.csv")

#importing the liberaries that w'll use
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('titanic/train.csv')


#Splitting data into train and test
X = data.drop('Survived',axis=1)
y = data['Survived']
X_train , X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2 )


#Exploring and transforming the data
df_train = X_train
df_test = X_test


def drop_col(data):
    data = data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
    return data 

df_train = drop_col(df_train)

df_test = drop_col(df_test)


medfair = df_train.describe()['Fare']['50%']

df_train.groupby('Sex')['Age'].median()

f_mean = df_train.groupby('Sex')['Age'].mean()[0]
m_mean = df_train.groupby('Sex')['Age'].mean()[1]

def fillnull(data):
    
    data.loc[data[data['Sex'] == 'female'].index,'Age'] = data.loc[data[data['Sex'] == 'female'].index,'Age'].fillna(f_mean)
    data.loc[data[data['Sex'] == 'male'].index,'Age'] = data.loc[data[data['Sex'] == 'male'].index,'Age'].fillna(m_mean)
    data['Fare'] = data['Fare'].fillna(medfair)
    return data

df_train = fillnull(df_train)

df_test = fillnull(df_test)



def clp_fare(data):
    data['Fare'] = data['Fare'].apply(lambda x: 200 if x>200 else x)
    return data

df_train  = clp_fare(df_train)

df_test = clp_fare(df_test)


#Preprocessing and training models
from sklearn.preprocessing import LabelEncoder

def encoding_categorical(data):
    le = LabelEncoder()
    cols = ['Sex','Embarked']
    for i in cols:
        data[i] = le.fit_transform(data[i])
    return data 

df_train = encoding_categorical(df_train)

df_test = encoding_categorical(df_test)

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

df_train = scaler.fit_transform(df_train)
df_test = scaler.fit_transform(df_test)

#### Modeling

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier()

clf1.fit(df_train,y_train)
clf2.fit(df_train,y_train)
clf3.fit(df_train,y_train)



#Testing

##### DecisionTree

print("decisionTree score : " ,clf1.score(df_test,y_test))

#### RandomForest

print("RandomForest score : " ,clf2.score(df_test,y_test))

##### Adaboost

print("Adaboost score : " ,clf3.score(df_test,y_test))

 
