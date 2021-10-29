import os
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('R.csv')
ts = data['Scale']
n, bins, patches = plt.hist(x=ts, bins='auto',
                            alpha=1, rwidth=0.85)
plt.xlabel('Scale (micrometres)')
plt.ylabel('Frequency')
plt.title('Hisogram of Scale')
plt.show()

setnames = set(data['Name'])
occurence = pd.DataFrame(columns = setnames)
for i in list(setnames):
    occurence[i] = len(data.loc['Name' == i, 'Name'])