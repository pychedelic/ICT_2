import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

df = pd.read_csv('diabetes.csv')

digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
Normal = Normalizer()
knn = KNeighborsClassifier(n_neighbors= 3)
pipeline = make_pipeline(Normal,knn)

pipeline.fit(x_train,y_train)

y_pred = knn.predict(x_test)
print('Test set predicitons: \n', y_pred)
knn.score(x_test,y_test)

#knn.save_model('diabetest_t_t_weights.h5')

Normal = Normalizer()
model = Sequential()

model.add(Dense(100, activation = 'relu', input_shape = (x.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 1, validation_split = 0.2)
model.predict(x_test)
model.evaluate(x_test,y_test)
model.summary()