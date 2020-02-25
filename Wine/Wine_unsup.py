from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from keras.utils import to_categorical
#import numpy as np
from sklearn import datasets

df = pd.read_csv('wine.csv')
samples = df.values
varieties = ['alcohol', 'malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280','proline']
digits = datasets.load_digits()
x= digits.data
y = digits.target
#y = to_categorical

#print(df.head())

#samples = np.array(df)
scaler = StandardScaler()
model = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler,model)
pipeline.fit(samples)
labels = pipeline.predict(samples)
print(labels)
df_1 = pd.DataFrame({'labels': labels, 'varieties':varieties})
print(df_1.sort_values)

#ct = pd.crosstab(df['class_label'],df['class_name'])
#print(ct)


