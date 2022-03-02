#importing libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def preprocessing(data):
  #Label Encode
  le = LabelEncoder()
  data['Elevation']= le.fit_transform(data['Elevation'])
  data['Aspect']= le.fit_transform(data['Aspect'])
  data['Hillshade_9am']= le.fit_transform(data['Hillshade_9am'])
  data['Hillshade_Noon']= le.fit_transform(data['Hillshade_Noon'])
  data['Horizontal_Distance_To_Fire_Points']= le.fit_transform(data['Horizontal_Distance_To_Fire_Points'])
  data['Slope']= le.fit_transform(data['Slope'])
  #Scaling
  return data

def predict(test_set) :
  data= pd.read_csv("covtype_train.csv")
  X_test = pd.read_csv(test_set) 

  Y_train= data['target']
  X_train= data.drop(['target'],axis=1)
  X_train= preprocessing(X_train)

  X_test= preprocessing(X_test)
  #Principle component Analysis
  pca = PCA(random_state=0)
  X_train= pca.fit_transform(X_train)
  X_test= pca.transform(X_test)
  
  kmeans = KMeans(n_clusters=7)
  kmeans.fit(X_train)

  pred_train = kmeans.predict(X_train)
  pred_test = kmeans.predict(X_test)
  cm = confusion_matrix(Y_train, pred_train)
  cm_argmax = cm.argmax(axis=0)
  y_pred_test = np.array([cm_argmax[i] for i in pred_test])
  y_pred_test = list(y_pred_test)

  return y_pred_test