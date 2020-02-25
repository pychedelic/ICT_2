import pandas as pd
import numpy as np
from sklearn import datasets
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
samples = list(df.pop('diabetes'))
diab = df.values

width = diab[:,4]
length = diab[:,7]

plt.scatter(width,length)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(width,length)
print(correlation)

norm = Normalizer()
pca = PCA(n_components= 2)
kmeans = KMeans(n_clusters =2)
pipe = make_pipeline(norm,pca,kmeans)
pipe.fit(diab)
features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA features')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 42)

normal = Normalizer()
knn = KNeighborsClassifier(n_neighbors=3)
pipeline = make_pipeline(normal,knn)
pipeline.fit(x_train,y_train)
preds = pipeline.predict(x_test)
print(preds)
score = pipeline.score(x_test,y_test)
print(score)

normalizer = Normalizer()
model = Sequential()
model.add(Dense(20, activation = 'relu', input_shape = (x.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(10, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(5, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(y.shape[1], activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
training = model.fit(x_train,y_train,epochs = 15, validation_split= 0.2)
pred = model.predict(x_test)
print(pred)
eval = model.evaluate(x_test,y_test)
print(eval)
model.summary()

model_1 = Sequential()
model_1.add(Dense(1000, activation = 'relu', input_shape = (x.shape[1],)))
model_1.add(Dropout(0.2))
model_1.add(Dense(700, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(500, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(200, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(100, activation = 'relu'))
model_1.add(Dropout(0.2))
model.add(Dense(10,activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(y.shape[1], activation = 'softmax'))

model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training_1 = model_1.fit(x_train,y_train,epochs = 15, validation_split = 0.2)
pred_1 = model_1.predict(x_test)
print(pred_1)
eval_1 = model_1.evaluate(x_test,y_test)
print(eval_1)

plt.plot(training.history['val_acc'],'b', training_1.history['val_acc'],'r')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Score')
plt.show()
