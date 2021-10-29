import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


data = pd.read_csv('R.csv')
scalepix = data["Tensile Strength (Yield)"]     #data['Scale'].values/data['Scale Length (px)'].values
n, bins, patches = plt.hist(x=scalepix, bins='auto',
                            alpha=1, rwidth=0.85, color='red')
plt.xlabel('Tensile Yield Strength (MPa)')
plt.ylabel('Frequency')
plt.title('Histogram of Tensile Yield Strength in the Whole Dataset')
plt.show()

setnames = set(data['Name'])
occurence = pd.DataFrame(columns = setnames)
for i in list(setnames):
    occurence[i] = len(data.loc['Name' == i, 'Name'])