from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
def accuracy(y,y_pred):
    accuracyscore=accuracy_score(y,y_pred)
    return accuracyscore
    
def matrix(y,y_pred):    
    confusionmatrix=confusion_matrix(y,y_pred)
    return confusionmatrix

def precision(y,y_pred):    
    precisionscore=precision_score(y,y_pred,pos_label='Presence')
    return precisionscore

def recall(y,y_pred):
    recallscore=recall_score(y,y_pred,pos_label='Presence')
    return recallscore

def f1(y,y_pred):    
    f1score=f1_score(y,y_pred,pos_label='Presence')
    return f1score