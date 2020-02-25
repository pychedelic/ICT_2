import pandas as pd
from sklearn import datasets
from keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

df = pd.read_csv('TSLA.csv')
samples = list(df.pop('Date'))
tesla = df.values

feature_1 = tesla[:,1]
feature_2 = tesla[:,4]

plt.scatter(feature_1,feature_2)
plt.axis('equal')
plt.show()

correlation,pvalue = pearsonr(feature_1,feature_2)
print(correlation)

norm = Normalizer()
pca = PCA(n_components= 3)


pipe = make_pipeline(norm,pca)
pipe.fit(tesla)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA Features')
plt.ylabel('Variance')
plt.xticks(features)
plt.show()

digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state= 1)

normalize = Normalizer()
knn = KNeighborsClassifier(n_neighbors = 5)
pipeline = make_pipeline(normalize, knn)
pipeline.fit(x_train,y_train)
y_pred = pipeline.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

Normalize = Normalizer()
model = Sequential()
model.add(Dense(1000, activation = 'relu', input_shape = (x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(700, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training = model.fit(x_train,y_train, epochs = 15, validation_split = 0.2)
pred = model.predict(x_test)
print(pred)
eval = model.evaluate(x_test, y_test)
print(eval)

plt.plot(training.history['val_acc'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation Score')
plt.show()
