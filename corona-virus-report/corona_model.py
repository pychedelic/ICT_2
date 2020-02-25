import pandas as pd
from sklearn import datasets
from keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout, BatchNormalization
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from sklearn.manifold import TSNE

#Read data
df = pd.read_csv('corona_confirmed.csv')
samples = list(df.pop('Province/State'))
samples_1 = list(df.pop('Country/Region'))
corona = df.values

#Find the correlation between features
feature_1 = corona[:,1]
feature_2 = corona[:,9]
plt.scatter(feature_1,feature_2)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(feature_1,feature_2)
print(correlation)

#Identify the principal components
normal = Normalizer()
pca = PCA()
kmeans = KMeans(n_clusters = 2)
pipeline = make_pipeline(normal,pca,kmeans)
pipeline.fit(corona)

features = range(pca.n_components_)
plt.bar(features,pca.explained_variance_)
plt.xlabel('PCA Features')
plt.ylabel('Variance')
plt.xticks(features)
plt.show()

#Exploratory Data Analysis
digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

#splitting the dataset into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, stratify = y, random_state = 42)

#Creating a pipeline to normalize the data and calculate the knn score
normal_1 = Normalizer()
knn = KNeighborsClassifier(n_neighbors = 2)

pipeline_1 = make_pipeline(normal_1, knn)
pipeline_1.fit(x_train,y_train)
y_pred = pipeline_1.predict(x_test)
print(y_pred)
score = pipeline_1.score(x_test,y_test)
print(score)

#Creating a dendrogram  to identify the closely related regions
mergings = linkage(corona, method = 'complete')
dendrogram(mergings, labels = samples, leaf_rotation= 90, leaf_font_size= 15)
labels = fcluster(mergings, 2, criterion = 'distance')
print(labels)
plt.show()

#Creating a multilayer perceptron NN, compiling the model, training the model, predicting the y_labels and evaluating the test set
model = Sequential()
model.add(Dense(1000, activation = 'relu', input_shape= (x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(800, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(600, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training = model.fit(x_train, y_train, epochs = 20, validation_split= 0.2)

model_pred = model.predict(x_test)
print(model_pred)

eval = model.evaluate(x_test, y_test)
print(eval)

#Summarising the model
model.summary()

#Batch Normalization model
model_1 = Sequential()
model_1.add(Dense(500, activation = 'relu', input_shape = (x.shape[1],)))
model_1.add(BatchNormalization())
model_1.add(Dense(250, activation = 'relu'))
model_1.add(BatchNormalization())
model_1.add(Dense(100, activation = 'relu'))
model_1.add(BatchNormalization())
model_1.add(Dense(y.shape[1], activation = 'softmax'))


model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training_1 = model_1.fit(x_train, y_train, epochs = 20, validation_split = 0.2)
pred_1 = model_1.predict(x_test)
print(pred_1)

eval_1 = model_1.evaluate(x_test, y_test)
print(eval_1)

model_1.summary()


#plotting the val_acc
plt.plot(training.history['val_acc'],'b', training_1.history['val_acc'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

#Creating a TSNE model
model_1 = TSNE(learning_rate= 150, perplexity= 30)
transformed = model_1.fit_transform(corona)
xs = transformed[:,0]
ys = transformed[:,1]

#Labelling
plt.scatter(xs,ys,alpha = 0.5)
for a, b, c_virus in zip(xs, ys, samples):
    plt.annotate(c_virus, (a, b), fontsize=5, alpha=0.75)
plt.show()
plt.show()

#Illustrate the feedforward network model
plot_model(model, to_file = 'corona_model.png')
data = plt.imread('corona_model.png')
plt.imshow(data)
plt.show()
