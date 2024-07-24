from sklearn.svm import SVC
import pickle
def model(x,y):
    svm=SVC()
    svm.fit(x,y)
    pickle.dump(svm,open(r'C:\Users\nirik\myfiles\myprojects\machine learning\heart disease prediction\models\model.pkl','wb'))
    return svm