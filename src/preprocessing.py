import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df=df.dropna()
    df.to_csv(r'C:\Users\nirik\myfiles\myprojects\machine learning\heart disease prediction\data\processed\cleaned_Heart_Disease_dataset.csv')
    return df

"""def label_encode(target):
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    encoded_target=le.fit_transform(target)
    return encoded_target"""

def split_data(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
    return x_train,x_test,y_train,y_test