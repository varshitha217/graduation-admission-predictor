import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path):
    df=pd.read_csv(path)
    df['gender']=df['gender'].map({'M':1,'F':0})
    X=df[['gpa','entrance_score','projects','internships','gender','age']]
    y=df['admitted']
    return train_test_split(X,y,test_size=0.2,random_state=42)
