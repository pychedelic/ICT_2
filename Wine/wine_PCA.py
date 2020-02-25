import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

df = pd.read_csv('wine.csv')
contents = list(df.pop('class_label'))
wine = df.values

model = PCA()

pca_features = model.fit_transform(wine)

xs = pca_features[:,0]
ys = pca_features[:,3]

plt.scatter(xs,ys)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(xs,ys)
print(correlation)