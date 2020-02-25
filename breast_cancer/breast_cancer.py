import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets



df = pd.read_csv('Breast_cancer_data.csv')
predictions = df.drop(['diagnosis'], axis = 1).as_matrix()
n_cols = predictions.shape[1]
input_data = (n_cols,)
target = to_categorical(df.diagnosis)

digits = datasets.load_digits()
x= digits.data
y = digits.target


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 21)


def bc_model(learning_rate = 0.01, activation = 'relu', optimizer = 'adam'):
    opt = optimizer(lr = learning_rate)
    model = Sequential()
    model.add(Dense(128, input_shape=(30,), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create a KerasClassifier
model = KerasClassifier(build_fn = bc_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32,128,256],
          'epochs': [50,100,200], 'learning_rate': [0.1,0.01,0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = 3)

# Running random_search.fit(X,y) would start the search,but it takes too long!
random_search.fit(x,y)