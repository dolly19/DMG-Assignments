!pip install pyclustering
#importing libraries
import numpy as np
from sklearn.metrics import f1_score
import pickle
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot
from numpy import unique
from numpy import where
from sklearn.preprocessing import StandardScaler, normalize

#reading data
data = pd.read_csv("covtype_train.csv") 

#Data Analysis
print("Data Description: ")
print(data.info())
print("\n")

#null values
print(data.isnull().sum())

"""#Pre-Processing:"""

#Label Encode
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Elevation']= le.fit_transform(data['Elevation'])
data['Aspect']= le.fit_transform(data['Aspect'])
data['Hillshade_9am']= le.fit_transform(data['Hillshade_9am'])
data['Hillshade_Noon']= le.fit_transform(data['Hillshade_Noon'])
data['Horizontal_Distance_To_Fire_Points']= le.fit_transform(data['Horizontal_Distance_To_Fire_Points'])
data['Slope']= le.fit_transform(data['Slope'])

#Scaling 
from sklearn.preprocessing import MinMaxScaler
#splitting into features and labels
Y= data['target']
X= data.drop(['target'],axis=1)
column_name= X.columns
# Scaling the Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled , columns = column_name)

"""#Feature Transformation:"""

#Principle component Analysis
pca = PCA(random_state=0)
X= pca.fit_transform(X)
print(X.shape)

"""#Question 1:"""

#Function for visulaization
def visualization(model, model_name):
  X_ran1, X_ran2, Y_ran1, Y_ran2 = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=123)
  pred = model.predict(X_ran2)

  #Visualization 1
  area = (12 * np.random.rand(len(X_ran2)))**2
  fig = plt.figure(figsize=(22,10))
  ax1 = fig.add_subplot(1,2,1,projection ="3d")
  ax2 = fig.add_subplot(1,2,2,projection ="3d")
  ax1.set_title(model_name+'Clustering Model')
  scatter = ax1.scatter3D(X_ran2[:, 0], X_ran2[:, 1], X_ran2[:, 2], s=area, c=pred)
  legend1 = ax1.legend(*scatter.legend_elements(),loc="upper right", title="Classes")
  ax1.add_artist(legend1)
  ax2.set_title('Original Data')
  scatter2 = ax2.scatter3D(X_ran2[:, 0], X_ran2[:, 1], X_ran2[:, 2], s=area, c=Y_ran2)
  legend2 = ax2.legend(*scatter2.legend_elements(),loc="upper right", title="Classes")
  ax2.add_artist(legend2)
  plt.show()

#Function to Compare your cluster distribution with the true label count.
def compare_distribution(model):
  pred= model.predict(X)
  unique_c, counts = np.unique(pred, return_counts=True)

  df = pd.DataFrame(pred, columns=['predicted'])
  for i in range(0, 7):
    print("Percentage of true labels in Cluster ", str(i))
    print('Total Instances : ', counts[i])
    df_value= df.loc[df['predicted'] == i]
    df_index= df_value.index
    true_value = Y[df_index]
    print(true_value.value_counts(normalize=True))
    print("\n")

  print('Total True Label Count :', )
  print(Y.value_counts().sort_index())

"""##Gaussian based clustering modelling:"""

from sklearn.mixture import GaussianMixture
model_gaus = GaussianMixture(n_components=7)
model_gaus.fit(X)

#representative object of each cluster
rep_gaus= model_gaus.means_
rep_gaus= pd.DataFrame(rep_gaus)
rep_gaus.head(7)

#Visualization of the clusters
visualization(model_gaus, "GaussianMixture " )

#Compare your cluster distribution with the true label count.
compare_distribution(model_gaus)

"""##K Median Clustering:"""

from pyclustering.cluster.kmedians import kmedians
intial = X[0:7]
kmedian = kmedians(X, intial)
kmedian.process()

#representative object of each cluster
rep_kmedian = kmedian.get_medians()
rep_kmedian= pd.DataFrame(rep_kmedian)
rep_kmedian.head(7)

#Visualization of the clusters
visualization(kmedian, "K Median " )

#Compare your cluster distribution with the true label count.
compare_distribution(kmedian)

"""##K-Means Clustering:"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7)
kmeans.fit(X)

#representative object of each cluster
rep_kmeans = kmeans.cluster_centers_
rep_kmeans = pd.DataFrame(rep_kmeans)
rep_kmeans.head(7)

#Visualization of the clusters
visualization(kmeans, "K Means " )

#Compare your cluster distribution with the true label count.
compare_distribution(kmeans)

"""##BIRCH Clustering:"""

from sklearn.cluster import Birch
model_birch = Birch(threshold=0.01, n_clusters=7)
model_birch.fit(X)

#representative object of each cluster
rep_birch = model_birch.subcluster_centers_
rep_birch  = pd.DataFrame(rep_birch)
rep_birch.head(7)

#Visualization of the clusters
visualization(model_birch, "BIRCH " )

#Compare your cluster distribution with the true label count.
compare_distribution(model_birch)

"""##Part 4: Compare the cluster formation of the gaussian based method with the other three clustering"""

result_gaus= model_gaus.predict(X)

results=[]
results.append(result_gaus)
results.append(kmedian.predict(X))
results.append(kmeans.predict(X))
results.append(model_birch.predict(X))

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

nmi_results = []
ars_results = []

y_true_val = list(result_gaus)

# Append the results into lists
for y_pred in results:
    nmi_results.append(normalized_mutual_info_score(y_true_val, y_pred))
    ars_results.append(adjusted_rand_score(y_true_val, y_pred))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 5))

x = np.arange(4)
avg = [sum(x) / 2 for x in zip(nmi_results, ars_results)]

xlabels = ['Gaussian', 'K Meadian', 'K Means', 'Birch']

sns.barplot(x, nmi_results, palette='Blues', ax=ax1)
sns.barplot(x, ars_results, palette='Reds', ax=ax2)
sns.barplot(x, avg, palette='Greens', ax=ax3)

ax1.set_ylabel('NMI Score')
ax2.set_ylabel('ARS Score')
ax3.set_ylabel('Average Score')

# # Add the xlabels to the chart
ax1.set_xticklabels(xlabels)
ax2.set_xticklabels(xlabels)
ax3.set_xticklabels(xlabels)

# Add the actual value on top of each bar
for i, v in enumerate(zip(nmi_results, ars_results, avg)):
    ax1.text(i - 0.1, v[0] + 0.01, str(round(v[0], 2)))
    ax2.text(i - 0.1, v[1] + 0.01, str(round(v[1], 2)))
    ax3.text(i - 0.1, v[2] + 0.01, str(round(v[2], 2)))

# Show the final plot
print("NMI , ARS and Average Scores of all clustering algorithms as compared to Gaussian Clustering:")
plt.title("NMI , ARS and Average Scores of all clustering algorithms as compared to Gaussian Clustering")
plt.show()

"""#Question 2:"""

#splitting into training and testing data sets in 8:2 ratio
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, stratify=data['target'], random_state=123)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7)
kmeans.fit(X_train)

pred_train= kmeans.predict(X_train)
pred_val= kmeans.predict(X_val)
cm = confusion_matrix(Y_train, pred_train)
cm_argmax = cm.argmax(axis=0)
y_pred_val = np.array([cm_argmax[i] for i in pred_val])
score= f1_score(Y_val,y_pred_val, average= 'micro')
print("Balanced F1- score : ", score )