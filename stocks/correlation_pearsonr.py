import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
df = pd.read_csv('stock.csv')
companies = list(df.pop('companies'))
samples = df.values

width = samples[:,0]
length = samples[:,1]

plt.scatter(width,length)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(width,length)
print(correlation)
