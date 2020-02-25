import pandas as pd
import numpy as np
from sklearn import datasets
from keras.utils import to_categorical, plot_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv('Breast_cancer_data.csv')
samples = list(df.pop('diagnosis'))
bc = df.values
#----------------------------------------------------------------------------------------------------------------------#
width = bc[:,0]
length = bc[:,1]

plt.scatter(width,length)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(width,length)
print(correlation)
#----------------------------------------------------------------------------------------------------------------------#

normal = Normalizer()
pca = PCA()

kmeans = KMeans(n_clusters= 2)

pipeline = make_pipeline(normal,pca, kmeans)

pipeline.fit(bc)

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA features')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
#----------------------------------------------------------------------------------------------------------------------#
digits = datasets.load_digits()
x= digits.data
y = digits.target
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state =42)

normalizer = Normalizer()
knn = KNeighborsClassifier(n_neighbors = 3)

pipe = make_pipeline(normalizer,knn)
pipe.fit(x_train,y_train)
preds = pipe.predict(x_test)
print(preds)
score = pipe.score(x_test,y_test)
print(score)
#----------------------------------------------------------------------------------------------------------------------#
model = Sequential()
model.add(Dense(20, activation = 'relu', input_shape = (x.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(10, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(5,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(y.shape[1],activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training = model.fit(x_train,y_train,epochs =15, validation_split = 0.2)
pred = model.predict(x_test)
print(pred)
eval = model.evaluate(x_test,y_test)
print(eval)
model.summary()

model_1 = Sequential()
model_1.add(Dense(1000, activation = 'relu', input_shape = (x.shape[1],)))
model_1.add(Dropout(0.2))
model_1.add(Dense(500, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(250, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(100, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(50,activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(10,activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(y.shape[1], activation = 'softmax'))

model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training_1 = model_1.fit(x_train,y_train,epochs = 15, validation_split = 0.2)
pred_1 = model_1.predict(x_test)
print(pred_1)
eval_1 = model_1.evaluate(x_test,y_test)
print(eval_1)
model_1.summary()
#----------------------------------------------------------------------------------------------------------------------#
plt.plot(training.history['val_acc'],'b',training_1.history['val_acc'],'r')
plt.title("Accuracy comparison of models")
plt.xlabel('Epochs')
plt.ylabel('Validation Score')
plt.show()