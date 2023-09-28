# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values.


## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRADEEPASRI S
RegisterNumber: 212221220038

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
print("Placement data:")
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or col
print("Salary data:")
data1.head()

print("Checking the null() function:")
data1.isnull().sum()

print ("Data Duplicate:")
data1.duplicated().sum()

print("Print data:")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

print("Data-status value of x:")
x=data1.iloc[:,:-1]
x

print("Data-status value of y:")
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print ("y_prediction array:")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") #A Library for Large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score =(TP+TN)/
#accuracy_score(y_true,y_pred,normalize=False)
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
print("Confusion array:")
confusion

from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("Classification report:")
print(classification_report1)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/75702726-2ea0-46aa-8b1b-b9c0b45dd48c)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/21de031d-d030-4d78-a2c2-867ff2803c98)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/c5f5e17a-9bc4-499f-b438-9a48f9a19b75)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/101b5fd8-9e85-4e36-89a0-d74298c6c5b2)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/3cfd9744-c003-44e9-b10a-4dbc753dbf48)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/3e32abc2-3b0a-4cb8-b0cd-d4b530740ab9)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/3c882578-47fc-4477-878f-28a4c976abb6)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/fc49b1ed-db84-4c4b-89a8-e55f80c4f333)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/2b8ad721-8163-459d-9652-b1170b6f301f)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/90486a86-1435-4dfd-89f2-5740f1eabde9)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/a5861266-578f-44a9-8f7b-d2916cb6350e)
![image](https://github.com/pradeepasri26/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433142/ceade70f-39d8-407d-8db2-3799616afe84)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
