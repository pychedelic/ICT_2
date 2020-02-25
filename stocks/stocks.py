from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from scipy.cluster.hierarchy import linkage,fcluster, dendrogram
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

df = pd.read_csv('stock.csv')
companies = list(df.pop('companies'))
samples= df.values

digits = datasets.load_digits()
x= digits.data
y = digits.target
y = to_categorical(y)

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

Normal = Normalizer()
model = KMeans(n_clusters = 6)
pipeline = make_pipeline(Normal,model)
pipeline.fit(x_train,y_train)
labels = pipeline.predict(x_test)
#pipeline.fit(y)
#companies = pipeline.predict(y)
print(labels)
score = pipeline.score(x_test,y_test)
print(score)

mergings = linkage(samples, method = 'complete')
dendrogram(mergings, labels =companies, leaf_rotation = 90, leaf_font_size = 6)
labels_1 = fcluster(mergings, 15, criterion='distance')
print(labels_1)

#print(companies)
#df_1 = pd.DataFrame({'labels':labels, 'companies': companies})
#ct = pd.crosstab(df_1['labels'], df_1['companies'])
#print(df_1.sort_values)
#print(ct)
plt.show()

#TSNE
model_1 = TSNE(learning_rate= 50)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]

#Labelling
plt.scatter(xs,ys,alpha = 0.5)
for a, b, company in zip(xs, ys, companies):
    plt.annotate(company, (a, b), fontsize=5, alpha=0.75)
plt.show()
plt.show()