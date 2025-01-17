import pandas as pd
from sklearn import datasets
from keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout, BatchNormalization
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from sklearn.manifold import TSNE


df = pd.read_csv('Dataset_spine.csv')
samples = list(df.pop('Class_att'))
spine = df.values

feature_1 = spine[:,0]
feature_2 = spine[:,2]
plt.scatter(feature_1, feature_2)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(feature_1,feature_2)
print(correlation)

normal = Normalizer()
pca = PCA(n_components = 2)
pipeline = make_pipeline(normal,pca)
pipeline.fit(spine)

features = range(pca.n_components_)
plt.bar(features,pca.explained_variance_)
plt.xlabel('PCA Features')
plt.ylabel('Variance')
plt.xticks(features)
plt.show()

digits = datasets.load_digits()
x = digits.data
y = digits.target
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state= 42)

normal_1 = Normalizer()
knn = KNeighborsClassifier(n_neighbors = 2)
pipe = make_pipeline(normal_1, knn)
pipe.fit(x_train, y_train)

pred = pipe.predict(x_test)
print(pred)

score = knn.score(x_test,y_test)
print(score)

model = Sequential()
model.add(Dense(1000, activation = 'relu', input_shape=(x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(800, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(600, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training = model.fit(x_train, y_train, epochs = 20, validation_split = 0.2)
pred = model.predict(x_test)
print(pred)
eval = model.evaluate(x_test,y_test)
print(eval)

model.summary()

plt.plot(training.history['val_acc'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Variance')
plt.show()

#Creating a dendrogram  to identify the closely related regions
mergings = linkage(spine, method = 'complete')
dendrogram(mergings, labels = samples, leaf_rotation= 90, leaf_font_size= 15)
labels = fcluster(mergings, 2, criterion = 'distance')
print(labels)

#Creating a TSNE model
model_1 = TSNE(learning_rate= 150, perplexity= 5)
transformed = model_1.fit_transform(spine)
xs = transformed[:,0]
ys = transformed[:,1]

#Labelling
plt.scatter(xs,ys,alpha = 0.5)
for a, b, pain in zip(xs, ys, samples):
    plt.annotate(pain, (a, b), fontsize=5, alpha=0.75)
plt.show()
plt.show()
