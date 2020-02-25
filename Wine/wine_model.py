import pandas as pd
from sklearn import datasets
from keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from keras.utils import to_categorical,plot_model
from sklearn.model_selection import train_test_split
from keras. layers import Dense, BatchNormalization, Dropout

df = pd.read_csv('wine.csv')
samples = list(df.pop('class_label'))
wine = df.values

normal = Normalizer()
pca = PCA()

kmeans = KMeans(n_clusters= 2)

pipeline = make_pipeline(normal,pca, kmeans)

pipeline.fit(wine)

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA features')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
#----------------------------------------------------------------------------------------------------------------------#

digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

normal_1 = Normalizer()

knn = KNeighborsClassifier(n_neighbors = 4)

pipeline_1 = make_pipeline(normal_1,knn)

pipeline_1.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(y_pred)
knn.score(x_test,y_test)

#----------------------------------------------------------------------------------------------------------------------#

model = Sequential()

model.add(Dense(100, activation = 'relu', input_shape = (x.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(50, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(25, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training = model.fit(x_train, y_train, epochs = 2, validation_split = 0.2)
pred = model.predict(x_test)
print(pred)
eval = model.evaluate(x_test,y_test)
print(eval)
model.summary()

model_1 = Sequential()
model_1.add(Dense(200, activation = 'relu', input_shape = (x.shape[1],)))
model_1.add(Dropout(0.2))
model_1.add(Dense(150, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(75, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(25, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(y.shape[1], activation = 'softmax'))

model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training_1 = model_1.fit(x_train,y_train,epochs = 2, validation_split = 0.2)
pred_1 = model_1.predict(x_test)
print(pred)
eval_1 = model.evaluate(x_test,y_test)
print(eval_1)
model_1.summary()

plt.plot(training.history['val_loss'],'b', training_1.history['val_loss'],'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

plot_model(model_1, to_file='wine_model.png')
data = plt.imread('wine_model.png')
plt.imshow(data)
plt.show()
