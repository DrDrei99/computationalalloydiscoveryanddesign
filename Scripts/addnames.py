import pandas as pd

df = pd.read_csv('temp.csv')
start = len(df.loc[~(df['Image'] == 'temp')])
for i in range(start,start+len(df.loc[df['Image'] == 'temp'])):
    name = df.iloc[i]['Name']
    prevname = df.iloc[i-1]['Name']
    if name != prevname:
        count = 0
    extension = '.png'
    image = name + '-' + str(count+1) + extension
    df['Image'][i] = image
    count = count + 1

df.to_csv('temp2.csv')

pass