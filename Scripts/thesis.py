import os
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('properties.csv')
res = data['Resolution'].values/1000000
n, bins, patches = plt.hist(x=res, bins='auto',
                            alpha=1, rwidth=0.85, color='black')
plt.xlabel('Image Resolution (Megapixels)')
plt.ylabel('Frequency')
plt.title('Histogram of Image Resolution')
plt.show()

setnames = set(data['Name'])
occurence = pd.DataFrame(columns = setnames)
for i in list(setnames):
    occurence[i] = len(data.loc['Name' == i, 'Name'])