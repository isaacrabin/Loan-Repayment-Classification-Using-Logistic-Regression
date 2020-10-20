#importing modules
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
#%matplotlib inline

#Loading the dataset
df = pd.read_csv("dataset\loan_data_set.csv")

#Data Pre-processing
#Filling in the missing numerical terms values
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean()) 
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean()) 
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

#Filling inthe categorical terms using the mode operation
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

#Exploratory Data Analysis
#Applying log function to the unevenly skewed attributes 
df['ApplicantIncome'] = np.log(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.log(df['CoapplicantIncome'])
df['LoanAmount'] = np.log(df['LoanAmount'])
df['Loan_Amount_Term'] = np.log(df['Loan_Amount_Term'])

#Creating a new attribute
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

#Applying transformation to the Attributes
df['ApplicantIncome'] = np.log(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.log(df['CoapplicantIncome'])
df['Loan_Amount_Term'] = np.log(df['Loan_Amount_Term'])
df['Total_Income'] = np.log(df['Total_Income'])

#Coorelation Matrix
corr = df.corr()

#Dropping the Unnecessari Columns
cols = ['CoapplicantIncome','Loan_ID','Total_Income']
df = df.drop(columns=cols, axis=1)

#LabelEncoding of the Categorical Features
from sklearn.preprocessing import LabelEncoder
cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

#Training the Attributes
X = df.drop(columns=['Loan_Status'],axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Model Training
#Creating a reusable classify class to use during the training
from sklearn.model_selection import cross_val_score
def classify(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
    model.fit(X_train,y_train) 
    print("Accuracy is:", model.score(X_test,y_test)*100)
    
    #Cross validation for the better validation of the model
    #It splits the data into several parts
    score = cross_val_score(model,X,y,cv=5)
    print("Cross validation is",np.mean(score)*100)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model,X,y)

#Confusion Matrix
model = LogisticRegression()
model.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

#pickling the model
import pickle
from sklearn import preprocessing
import joblib
import warnings
warnings.filterwarnings('ignore')
#filename = 'modelfinal.pkl'
#joblib.dump(model,filename)
#pickle.dump(model.open('mymodel.pkl','wb'))
with open('mymodel.pkl','wb') as pickle_file:
    pickle.dump(model,pickle_file)
