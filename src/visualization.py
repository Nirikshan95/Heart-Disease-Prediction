import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df,target):
    plt.figure(figsize=(8,6))
    sns.pairplot(df,hue=target)
    plt.suptitle('pairplot of Heart Disease Dataset')
    plt.savefig(fname=r'C:\Users\nirik\myfiles\myprojects\machine learning\heart disease prediction\reports\figures\features_pairplot.jpg')
    plt.show()
    
def plot_pca_data(features,target):
    data=pd.DataFrame(features,columns=['principal component 1','principal component 2','principal component 3','principal component 4'])
    data['target']=target
    
    plt.figure(figsize=(8,6))
    sns.pairplot(data)
    plt.suptitle('PCA of Heart Disease Dataset')
    plt.savefig(fname=r'C:\Users\nirik\myfiles\myprojects\machine learning\heart disease prediction\reports\figures\PCA_features_pairplot.jpg')
    plt.show()

def heat_map(x):
    sns.heatmap(x,cmap='Greens',annot=True)
    plt.title("Confusion Matrix")
    plt.savefig(fname=r'C:\Users\nirik\myfiles\myprojects\machine learning\heart disease prediction\reports\figures\confusion_matrx.jpg')
    plt.show()
    
def countplt(dataframe,column):
    cnt=dataframe[column]
    sns.countplot(cnt)
    plt.title(f"Connt Plot of {column}")
    plt.savefig(fname=fr'C:\Users\nirik\myfiles\myprojects\machine learning\heart disease prediction\reports\figures\cuuntplt_{column}.jpg')    
    plt.show()