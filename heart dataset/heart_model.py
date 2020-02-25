import pandas as pd
import numpy as np
from sklearn import datasets
from keras.utils import to_categorical, plot_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout

df = pd.read_csv('heart.csv')

digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


normal = Normalizer()
knn = KNeighborsClassifier(n_neighbors=3)

pipe = make_pipeline(normal,knn)
pipe.fit(x_train,y_train)

score = pipe.score(x_test,y_test)
print(score)


model = Sequential()
model.add(Dense(100, activation= 'relu', input_shape = (x.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(50, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
training = model.fit(x_train,y_train,epochs = 2, validation_split = 0.2)
pred = model.predict(x_test)
print(pred)
eval = model.evaluate(x_test,y_test)
print(eval)
model.summary()

model_1 = Sequential()
model_1.add(Dense(1000, activation='relu', input_shape = (x.shape[1],)))
model_1.add(Dropout(0.2))
model_1.add(Dense(500, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(250, activation = 'relu' ))
model_1.add(Dropout(0.2))
model_1.add(Dense(100, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(50, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(25, activation = 'relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(y.shape[1], activation = 'softmax'))

model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training_1 = model_1.fit(x_train,y_train, epochs = 15, validation_split= 0.2)
pred_1 = model_1.predict(x_test)
print(pred_1)
eval_1 = model_1.evaluate(x_test,y_test)
print(eval_1)
model_1.summary()

plt.plot(training.history['val_acc'],'b', training_1.history['val_acc'],'r')
plt.title('Model Comparison for Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()