from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('wine.csv')
varieties = list(df.pop('wine'))

samples = df.values
varieties = ['alcohol', 'malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280','proline']

mergings = linkage(samples,method = 'complete')
dendrogram(mergings,
           labels = varieties,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()